"""Exportación del universo sintético al FORMATO de CNEL EP (SIGELEC/ArcFM).

Escribe las capas con los **nombres y campos reales** de la geodatabase de CNEL,
de modo que el dataset de prueba recorra exactamente la misma ruta de ingesta que
los datos de producción (`lossan ingest-cnel`) y no un atajo.

Cubre la jerarquía completa:
    EstructuraSoporte ─┬─ PuestoTransfDistribucion ── UNIDADTRANSFDISTRIBUCION
                       └─ PuntoCarga ── CONEXIONCONSUMIDOR ── ATRIBUTOSCONSUMIDOR
"""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..domain.enums import BankConfig
from ..lakehouse import Lakehouse

# Dominio inverso: BankConfig -> texto que usa CNEL en CONFIGURACIONLADOBAJA
_BANK_TEXT = {
    BankConfig.SINGLE.value: "MONOFASICO",
    BankConfig.WYE_CLOSED.value: "ESTRELLA CERRADA",
    BankConfig.DELTA_CLOSED.value: "DELTA CERRADA",
    BankConfig.OPEN_DELTA.value: "DELTA ABIERTA",
    BankConfig.OPEN_WYE_OPEN_DELTA.value: "ESTRELLA ABIERTA DELTA ABIERTA",
    BankConfig.DELTA_4WIRE.value: "DELTA 4 HILOS",
    BankConfig.INDEPENDENT.value: "INDEPENDIENTE",
}
# Phase Designation de ArcFM (bitmask): A=4, B=2, C=1
_PHASE_CODE = {"A": 4, "B": 2, "C": 1, "AB": 6, "AC": 5, "BC": 3, "ABC": 7}
_TARIFF_TEXT = {"residential": "RESIDENCIAL", "commercial": "COMERCIAL",
                "industrial": "INDUSTRIAL", "streetlight_led": "ALUMBRADO PUBLICO",
                "streetlight_magnetic": "ALUMBRADO PUBLICO"}


def _guid(prefix: str, i: int, scope: str = "") -> str:
    """GUID determinístico con el formato de ArcGIS ``{...}``.

    ``scope`` (el alimentador) entra en la semilla: sin él, dos alimentadores
    generarían los mismos GUID y las relaciones padre-hijo se cruzarían.
    """
    return "{" + str(uuid.uuid5(uuid.NAMESPACE_OID,
                                f"{scope}-{prefix}-{i}")).upper() + "}"


def canonical_to_cnel(tables: dict[str, pd.DataFrame],
                      feeder_code: str) -> dict[str, pd.DataFrame]:
    """Traduce las tablas canónicas de un alimentador al formato CNEL."""
    poles = tables["poles"]; sites = tables["sites"]; units = tables["transformer_units"]
    lps = tables["load_points"]; cust = tables["customers"]
    sl = tables.get("streetlights"); dev = tables.get("switching_devices")

    # --- GUIDs estables por entidad ---
    pole_guid = {p: _guid("EST", i, feeder_code) for i, p in enumerate(poles["pole_id"])}
    site_guid = {s: _guid("PTD", i, feeder_code) for i, s in enumerate(sites["site_id"])}
    lp_guid = {l: _guid("PCG", i, feeder_code) for i, l in enumerate(lps["load_point_id"])}
    # CIRCUITSOURCEGUID del dispositivo aguas arriba (traza nativa ArcFM)
    dev_guid = ({d: _guid("CSD", i, feeder_code) for i, d in enumerate(dev["device_id"])}
                if dev is not None and not dev.empty else {})
    dev_list = list(dev_guid.values()) or [_guid("CSD", 0, feeder_code)]

    out: dict[str, pd.DataFrame] = {}

    # --- EstructuraSoporte (el poste) ---
    out["EstructuraSoporte"] = pd.DataFrame({
        "GLOBALID": [pole_guid[p] for p in poles["pole_id"]],
        "ALIMENTADORID": feeder_code,
        "TIPOESTRUCTURA": poles.get("pole_type", "POSTE HC"),
        "ALTURA": poles.get("height_m", 11.0),
        "COORD_X": poles["x"], "COORD_Y": poles["y"],
    })

    # --- PuestoTransfDistribucion ---
    rng = np.random.default_rng(abs(hash(feeder_code)) % (2 ** 32))
    site_cs = {s: dev_list[i % len(dev_list)] for i, s in enumerate(sites["site_id"])}
    site_own_cs = {s: _guid("CST", i, feeder_code) for i, s in enumerate(sites["site_id"])}
    out["PuestoTransfDistribucion"] = pd.DataFrame({
        "GLOBALID": [site_guid[s] for s in sites["site_id"]],
        "ALIMENTADORID": feeder_code,
        "CODIGOPUESTO": [f"PT-{feeder_code}-{i:04d}" for i in range(len(sites))],
        "ESTRUCTURASOPORTEGLOBALID": [pole_guid.get(p) for p in sites["pole_id"]],
        "CONFIGURACIONLADOBAJA": [_BANK_TEXT.get(b, "MONOFASICO")
                                  for b in sites["bank_config"]],
        "FASECONEXION": [_PHASE_CODE["ABC"]] * len(sites),
        "VOLTAJE": 13800,
        "POTENCIAKVA": np.nan,          # se recalcula desde las unidades
        "MEDIDO": 0,
        # CIRCUITSOURCEGUID: fuente de circuito PROPIA del puesto (la heredan
        # sus hijos); PARENTCIRCUITSOURCEGUID: el dispositivo aguas arriba.
        "CIRCUITSOURCEGUID": [site_own_cs[s] for s in sites["site_id"]],
        "PARENTCIRCUITSOURCEGUID": [site_cs[s] for s in sites["site_id"]],
    })

    # --- UNIDADTRANSFDISTRIBUCION (el banco) ---
    out["UNIDADTRANSFDISTRIBUCION"] = pd.DataFrame({
        "GLOBALID": [_guid("UTD", i, feeder_code) for i in range(len(units))],
        "PUESTOTRANSFDISTGLOBALID": [site_guid.get(s) for s in units["site_id"]],
        "ALIMENTADORID": feeder_code,
        "POTENCIANOMINAL": [f"{v:g} kVA" for v in units["sn_kva"]],   # texto de dominio
        "TENSIONLADOALTA": 13800,
        "FASECONEXION": [_PHASE_CODE.get(str(p), 7) for p in units["phase"]],
        "NUMEROSERIE": [f"SN{i:07d}" for i in range(len(units))],
        "MARCA": "ECUATRAN", "MODELO": "CONVENCIONAL", "TIPO": 1,
        "ESTADO": 1, "PROPIEDAD": "EMPRESA",
    })

    # --- PuntoCarga (el edificio/predio) ---
    out["PuntoCarga"] = pd.DataFrame({
        "GLOBALID": [lp_guid[l] for l in lps["load_point_id"]],
        "ALIMENTADORID": feeder_code,
        "PUESTOTRANSFDISTGLOBALID": [site_guid.get(s) for s in lps["transformer_site_id"]],
        "ESTRUCTURASOPORTEGLOBALID": [pole_guid.get(p) for p in lps["pole_id"]],
        "FASECONEXION": [_PHASE_CODE.get(str(p), 4) for p in lps["phase"]],
        "RUTALECTURA": [f"R{(i % 40) + 1:03d}" for i in range(len(lps))],
        "SECUENCIALECTURA": [str(i + 1) for i in range(len(lps))],
        "CODIGOCLIENTE": [f"CLI-{feeder_code}-{i:06d}" for i in range(len(lps))],
        "COORD_X": 0.0, "COORD_Y": 0.0,
        "PARENTCIRCUITSOURCEGUID": [site_own_cs.get(s) for s in lps["transformer_site_id"]],
    })

    # --- CONEXIONCONSUMIDOR (los medidores) ---
    n = len(cust)
    codigo_unico = [f"CU{feeder_code[-3:]}{i:07d}" for i in range(n)]
    cuenta_contrato = [f"CC{feeder_code[-3:]}{i:07d}" for i in range(n)]
    out["CONEXIONCONSUMIDOR"] = pd.DataFrame({
        "GLOBALID": [_guid("CXC", i, feeder_code) for i in range(n)],
        "CODIGOUNICO": codigo_unico,                      # identificador único SIG
        "PUNTOCARGAGLOBALID": [lp_guid.get(s) for s in cust["site_id"]],
        "ALIMENTADORID": feeder_code,
        "CODIGOCLIENTE": [f"CLI-{feeder_code}-{i:06d}" for i in range(n)],
        "MDENUMFAB": [f"MED{i:08d}" for i in range(n)],
        "MEDMAR": rng.choice(["ELSTER", "ITRON", "HEXING"], n),
        "TIPOMEDIDOR": rng.choice(["MONOFASICO", "BIFASICO", "TRIFASICO"],
                                  n, p=[0.8, 0.12, 0.08]),
        "SECUENCIAFASE": [_PHASE_CODE.get(str(p), 4) for p in cust["phase"]],
        "PREPAGO": "NO", "ESTADO": 1,
    })

    # --- ATRIBUTOSCONSUMIDOR (comercial: CUENTACONTRATO, tarifa, carga) ---
    out["ATRIBUTOSCONSUMIDOR"] = pd.DataFrame({
        "CODIGOUNICO": codigo_unico,                      # clave de unión
        "CUENTACONTRATO": cuenta_contrato,                # identificador comercial
        "CODIGOCLIENTE": [f"CLI-{feeder_code}-{i:06d}" for i in range(n)],
        "TIPOTARIFA": [_TARIFF_TEXT.get(t, "RESIDENCIAL") for t in cust["tariff_class"]],
        "CATEGORIA": rng.choice(["A", "B", "C"], n),
        "POTENCIAACTIVA": cust["installed_load_kw"].round(3),
        "POTENCIAREACTIVA": (cust["installed_load_kw"] * 0.35).round(3),
        "CDAFAS": [3 if str(p) == "ABC" else 1 for p in cust["phase"]],
        "EDCCOD": "A",
        "NUMMEDIDOR": [f"MED{i:08d}" for i in range(n)],
        "TIPOMEDIDOR": "MONOFASICO",
        "CONSUMOPROMEDIO": 0.0,           # se completa desde el consumo
        "ESTADO": "ACTIVO",
    })

    # --- Luminaria ---
    if sl is not None and not sl.empty:
        out["Luminaria"] = pd.DataFrame({
            "GLOBALID": [_guid("LUM", i, feeder_code) for i in range(len(sl))],
            "ALIMENTADORID": feeder_code,
            "ESTRUCTURASOPORTEGLOBALID": [pole_guid.get(p) for p in sl["site_id"]],
            "PARENTCIRCUITSOURCEGUID": [site_own_cs.get(s)
                                        for s in sl["transformer_site_id"]],
            "FASECONEXION": 4,
            "POTENCIA": sl["lamp_w"], "TECNOLOGIA": sl["technology"],
        })

    # --- Tramos (primario y secundario) ---
    segs = tables.get("segments")
    if segs is not None and not segs.empty:
        prim = segs[segs["section"] == "primary"]
        sec = segs[segs["section"] == "secondary"]
        if not prim.empty:
            out["TramoDistribucionAereo"] = pd.DataFrame({
                "GLOBALID": [_guid("TDA", i, feeder_code) for i in range(len(prim))],
                "ALIMENTADORID": feeder_code,
                "NODOINICIAL": prim["node_from"], "NODOFINAL": prim["node_to"],
                "CALIBRE": prim["conductor_code"], "LONGITUD": prim["length_m"],
                "FASECONEXION": [_PHASE_CODE.get(str(p), 7) for p in prim["phase"]],
                "VOLTAJE": prim["voltage_ll"], "MATERIAL": prim["material"],
                "RESISTENCIA": prim["r_ohm_per_km"], "REACTANCIA": prim["x_ohm_per_km"],
                "AMPACIDAD": prim["ampacity_a"],
            })
        if not sec.empty:
            out["TramoBajaTensionAereo"] = pd.DataFrame({
                "GLOBALID": [_guid("TBA", i, feeder_code) for i in range(len(sec))],
                "ALIMENTADORID": feeder_code,
                "NODOINICIAL": sec["node_from"], "NODOFINAL": sec["node_to"],
                "CALIBRE": sec["conductor_code"], "LONGITUD": sec["length_m"],
                "FASECONEXION": [_PHASE_CODE.get(str(p), 4) for p in sec["phase"]],
                "VOLTAJE": sec["voltage_ll"], "MATERIAL": sec["material"],
                "RESISTENCIA": sec["r_ohm_per_km"], "REACTANCIA": sec["x_ohm_per_km"],
                "AMPACIDAD": sec["ampacity_a"],
            })

    # --- PuestoProteccionDinamico (dispositivos / zonas) ---
    if dev is not None and not dev.empty:
        out["PuestoProteccionDinamico"] = pd.DataFrame({
            "GLOBALID": [dev_guid[d] for d in dev["device_id"]],
            "ALIMENTADORID": feeder_code,
            "CIRCUITSOURCEGUID": [dev_guid[d] for d in dev["device_id"]],
            "TIPO": dev["type"], "ESTADONORMAL": dev["normal_state"],
            "FASECONEXION": 7,
        })
    return out


def export_cnel_dataset(lake_root: str, out_dir: str, cfg: Config | None = None,
                        write_gpkg: bool = True) -> dict:
    """Escribe el universo del lakehouse como dataset CNEL (CSV + GeoPackage).

    Además genera el **consumo comercial por CUENTACONTRATO**, tal y como lo
    entrega el CIS, y la cabecera por alimentador.
    """
    cfg = cfg or load_config()
    lake = Lakehouse(lake_root)
    out = Path(out_dir)
    (out / "sig_csv").mkdir(parents=True, exist_ok=True)
    (out / "comercial").mkdir(parents=True, exist_ok=True)

    from ..pipeline.runner import list_feeders
    feeders = list_feeders(lake)
    all_layers: dict[str, list[pd.DataFrame]] = {}
    cc_by_customer: dict[str, str] = {}

    for fid in feeders:
        tables = {e: lake.read_entity("bronze", e, fid) for e in
                  ("poles", "sites", "transformer_units", "load_points",
                   "customers", "streetlights", "switching_devices", "segments")}
        if tables["customers"].empty:
            continue
        layers = canonical_to_cnel(tables, fid)
        for name, df in layers.items():
            all_layers.setdefault(name, []).append(df)
        # mapa customer_unit_id -> CUENTACONTRATO para el consumo comercial
        cc = layers["ATRIBUTOSCONSUMIDOR"]
        for orig, cuenta in zip(tables["customers"]["customer_unit_id"],
                                cc["CUENTACONTRATO"]):
            cc_by_customer[orig] = cuenta

    counts = {}
    for name, parts in all_layers.items():
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(out / "sig_csv" / f"{name}.csv", index=False)
        counts[name] = len(df)

    # --- consumo comercial POR CUENTACONTRATO (como lo entrega el CIS) ---
    cons = lake.read_entity("bronze", "consumption")
    if not cons.empty:
        com = pd.DataFrame({
            "CUENTA_CONTRATO": cons["customer_unit_id"].map(cc_by_customer),
            "PERIODO": cons["year_month"],
            "CONSUMO_KWH": cons["kwh"].round(2),
            "CONSUMO_KVARH": cons.get("kvarh", pd.Series(dtype=float)).round(2),
            "LECTURA_ESTIMADA": cons.get("estimated", False),
        }).dropna(subset=["CUENTA_CONTRATO"])
        com.to_csv(out / "comercial" / "consumo_historico.csv", index=False)
        counts["consumo_historico"] = len(com)

    header = lake.read_entity("bronze", "header_meters")
    if not header.empty:
        header.rename(columns={"feeder_id": "ALIMENTADOR", "year_month": "PERIODO",
                               "kwh": "ENERGIA_KWH", "kvarh": "ENERGIA_KVARH"}
                      ).to_csv(out / "comercial" / "cabecera_alimentador.csv", index=False)
        counts["cabecera"] = len(header)

    # verdad-terreno (solo para evaluar el modelo; no existe en producción)
    labels = lake.read_entity("bronze", "theft_labels")
    if not labels.empty:
        labels = labels.assign(
            cuenta_contrato=labels["customer_unit_id"].map(cc_by_customer))
        labels.to_csv(out / "verdad_terreno_hurtos.csv", index=False)
        counts["verdad_terreno"] = len(labels)

    # --- GeoPackage con las capas espaciales (abre en ArcGIS/QGIS) ---
    if write_gpkg:
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            est = pd.concat(all_layers["EstructuraSoporte"], ignore_index=True)
            g = gpd.GeoDataFrame(
                est, geometry=[Point(xy) for xy in zip(est["COORD_X"], est["COORD_Y"])],
                crs="EPSG:32717")           # UTM 17S (Ecuador)
            gpkg = out / "sig_cnel.gpkg"
            if gpkg.exists():
                gpkg.unlink()
            g.to_file(gpkg, layer="EstructuraSoporte", driver="GPKG")
            counts["gpkg"] = str(gpkg)
        except Exception:  # pragma: no cover
            pass
    return counts
