"""Regresiones de los defectos hallados en la revisión completa.

Cada test fija un bug concreto que estuvo presente y no debe reaparecer.
"""
import pandas as pd
import pytest

from lossan.config import load_config
from lossan.pipeline.balance import analyze_feeder
from lossan.synth.generator import SyntheticGenerator
from lossan.topology import FeederGraph


def _tables(micro_config):
    return SyntheticGenerator(load_config()).feeder_tables(0)


# --- BUG 1: la métrica de cierre era una identidad algebraica (siempre 0) ---

def test_balance_closure_is_not_tautological(micro_config):
    """El cierre debe DETECTAR un balance incoherente, no dar siempre 'ok'."""
    t = _tables(micro_config)
    cfg = load_config()
    ok = analyze_feeder(t, cfg, with_risk_proxy=False)["feeder_balance"].iloc[0]
    assert bool(ok["balance_coherent"]), ok["failed_checks"]

    # cabecera reducida a la mitad -> PNT negativa: DEBE fallar
    broken = dict(t)
    broken["header_meters"] = t["header_meters"].assign(
        kwh=t["header_meters"]["kwh"] * 0.5)
    bad = analyze_feeder(broken, cfg, with_risk_proxy=False)["feeder_balance"].iloc[0]
    assert not bool(bad["balance_coherent"])
    assert "pnt_non_negative" in bad["failed_checks"]

    # cabecera duplicada -> pérdidas totales implausibles: DEBE fallar
    inflated = dict(t)
    inflated["header_meters"] = t["header_meters"].assign(
        kwh=t["header_meters"]["kwh"] * 2.0)
    bad2 = analyze_feeder(inflated, cfg, with_risk_proxy=False)["feeder_balance"].iloc[0]
    assert not bool(bad2["balance_coherent"])
    assert "total_losses_plausible" in bad2["failed_checks"]


# --- BUG 2: alumbrado público repartido uniformemente entre todos los puestos ---

def test_streetlight_energy_assigned_by_trace(micro_config):
    """El AP debe ir al puesto de SU traza, no repartirse por igual (§10.2)."""
    t = _tables(micro_config)
    cfg = load_config()
    target = t["sites"]["site_id"].iloc[0]
    other = t["sites"]["site_id"].iloc[1]

    def loadability_of(streetlights):
        out = analyze_feeder(dict(t, streetlights=streetlights), cfg,
                             with_risk_proxy=False)["transformer_loadability"]
        return out.set_index("site_id")["s_max_kva"]

    # A: todas las luminarias en 'target' · B: todas en 'other'
    a = loadability_of(t["streetlights"].assign(transformer_site_id=target))
    b = loadability_of(t["streetlights"].assign(transformer_site_id=other))
    # El MISMO puesto debe cargar más cuando el AP le pertenece por traza.
    # Con el reparto uniforme anterior, a[target] == b[target] (bug).
    assert a[target] > b[target]
    assert b[other] > a[other]


# --- BUG 3: subtree_load identificaba hojas por substring del id ---

def test_subtree_load_uses_explicit_sets_not_name_matching(micro_config):
    t = _tables(micro_config)
    fg = FeederGraph.build("F0000", t["segments"], t["sites"],
                           t["customers"], t["streetlights"])
    total = fg.subtree_load(fg.source)
    assert total["n_customers"] == len(t["customers"])

    # con ids que no siguen ninguna convención, el conteo sigue siendo correcto
    fg2 = FeederGraph.build("F0000", t["segments"], t["sites"])
    assert fg2.subtree_load(fg2.source)["n_customers"] == 0   # sin registrar: 0, no adivina
    fg2.register_leaf_nodes(t["customers"], t["streetlights"])
    assert fg2.subtree_load(fg2.source)["n_customers"] == len(t["customers"])


# --- BUG 4: el generador nunca producía puestos de cliente multi-unidad ---

def test_generator_creates_multiunit_customer_sites(micro_config):
    t = _tables(micro_config)
    sizes = t["customers"].groupby("site_id").size()
    assert sizes.max() > 1, "sin puestos multi-unidad, M8 y §17.1 nunca se ejercitan"
    # las unidades de un mismo puesto comparten acometida: poste y transformador
    multi = sizes[sizes > 1].index[0]
    g = t["customers"][t["customers"]["site_id"] == multi]
    assert g["pole_id"].nunique() == 1
    assert g["transformer_site_id"].nunique() == 1


# --- BUG 5: R09 recorría el árbol por cada tramo (O(V·(V+E))) ---

def test_accumulate_downstream_matches_subtree_load(micro_config):
    t = _tables(micro_config)
    fg = FeederGraph.build("F0000", t["segments"], t["sites"], t["customers"])
    load_map = {n: 1.0 for n in list(fg.idx)[:50]}
    acc = fg.accumulate_downstream(load_map)
    # la acumulación en una pasada coincide con la traza nodo a nodo
    for node in list(fg.idx)[:25]:
        expected = fg.subtree_load(node, load_map)["load_kva"] + load_map.get(node, 0.0)
        assert acc[node] == pytest.approx(expected, abs=1e-6)


# --- BUG 6: parámetros de negocio hardcodeados (§22.13) ---

def test_no_hardcoded_business_constants_in_pipeline():
    """Factor de carga, fp y pérdidas de secundario deben venir de config."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src" / "lossan" / "pipeline"
    offenders = []
    for f in src.glob("*.py"):
        txt = f.read_text(encoding="utf-8")
        for pat in ("fc = 0.45", "SECONDARY_LOSS_PCT = ", "/ 0.92", "= 730.0"):
            if pat in txt:
                offenders.append(f"{f.name}: {pat}")
    assert not offenders, f"constantes de negocio en código: {offenders}"
    cfg = load_config()
    for key in ("default_load_factor", "default_site_pf", "secondary_loss_frac",
                "hours_per_month"):
        assert key in cfg.electrical


# --- BUG 7: el pipeline asumía datos que la GDB de CNEL no tiene -----------

def test_risk_model_without_ground_truth(micro_config):
    """En producción NO hay verdad-terreno: el modelo debe usar minería."""
    from lossan.ml import train_risk_model
    t = _tables(micro_config)
    # tabla vacía (lo que devuelve el lakehouse cuando la entidad no existe)
    for empty in (pd.DataFrame(), None):
        scores, info = train_risk_model(t["consumption"], t["customers"], empty,
                                        cfg=load_config())
        assert not scores.empty
        assert "risk_score" in scores.columns


def test_features_without_optional_sig_columns(micro_config):
    """La GDB de CNEL no trae service_drop_kva: no debe romper las features."""
    from lossan.ml import build_features
    t = _tables(micro_config)
    cust = t["customers"].drop(columns=["service_drop_kva"])
    feats = build_features(t["consumption"], cust)
    assert not feats.empty and "z_peer" in feats.columns


def test_transformer_catalog_fills_missing_losses():
    """La GDB solo trae kVA: P0/Pk se completan del catálogo y se marca origen."""
    from lossan.io.cnel import apply_transformer_catalog
    units = pd.DataFrame({"unit_id": ["u1", "u2", "u3"],
                          "sn_kva": [50.0, 37.5, 999.0]})   # 999 no está en catálogo
    out = apply_transformer_catalog(units)
    assert out["p0_kw"].notna().all() and out["pk_kw"].notna().all()
    assert set(out["plate_source"]) == {"catalog"}
    assert out.loc[0, "p0_kw"] == 0.145        # valor exacto del catálogo (50 kVA)
    assert out.loc[2, "p0_kw"] == pytest.approx(999.0 * 0.003, abs=1e-3)  # fracción


def test_validate_labels_without_ground_truth():
    from lossan.ml import validate_against_confirmed
    res = validate_against_confirmed(pd.DataFrame(), pd.DataFrame())
    assert res["confirmed"] == 0
