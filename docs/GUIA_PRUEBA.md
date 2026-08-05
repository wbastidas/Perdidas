# Guía de prueba paso a paso — dataset CNEL

Cómo ejecutar la plataforma completa, etapa por etapa, con el dataset de prueba
en **formato CNEL EP** (las mismas capas y campos de SIGELEC). Recorre la misma
ruta que los datos de producción, no un atajo.

## Contenido del dataset

```
dataset_cnel/
├── sig_csv/                        # capas del SIG con nombres/campos de CNEL
│   ├── EstructuraSoporte.csv           7.668  postes
│   ├── PuestoTransfDistribucion.csv      599  puestos de transformación
│   ├── UNIDADTRANSFDISTRIBUCION.csv    1.085  unidades (279 puestos con banco)
│   ├── PuntoCarga.csv                  8.126  puntos de carga
│   ├── CONEXIONCONSUMIDOR.csv         11.400  medidores (1.615 edificios multi)
│   ├── ATRIBUTOSCONSUMIDOR.csv        11.400  CUENTACONTRATO, tarifa, carga
│   ├── Luminaria.csv                   2.077  alumbrado público
│   ├── TramoDistribucionAereo.csv        599  tramos primarios
│   ├── TramoBajaTensionAereo.csv      11.400  tramos secundarios
│   └── PuestoProteccionDinamico.csv       46  dispositivos (zonas)
├── comercial/
│   ├── consumo_historico.csv         410.400  36 meses, por CUENTA_CONTRATO
│   └── cabecera_alimentador.csv          216  energía de cabecera
├── sig_cnel.gpkg                              capa espacial (ArcGIS/QGIS)
└── verdad_terreno_hurtos.csv          11.400  quién roba (solo para evaluar)
```

**6 alimentadores · 11.400 clientes · 36 meses.** Volumen suficiente para que
todas las etapas produzcan resultados con sentido estadístico y para que la
corrida tarde minutos, no horas.

> `verdad_terreno_hurtos.csv` **no existe en producción**: sirve para medir la
> precisión del modelo de detección contra la respuesta correcta.

---

## Paso 0 — Preparar el entorno

```bash
cd Perdidas
python -m venv .venv
source .venv/bin/activate           # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dashboard,dev,power,ml,geo,reports]"
pytest -q                           # debe pasar todo
```

Define dónde vivirán los datos procesados:

```bash
export LOSSAN_LAKEHOUSE=$PWD/data/lake        # Windows: $env:LOSSAN_LAKEHOUSE="$PWD\data\lake"
```

---

## Paso 1 — Ingesta del SIG (jerarquía puesto → unidad)

```bash
lossan ingest-cnel-csv dataset_cnel/sig_csv
```

Lee las capas de CNEL y las traduce al modelo canónico resolviendo las
relaciones por `GLOBALID`. **Qué mirar en la salida:**

```
ATRIBUTOSCONSUMIDOR: 11400/11400 conexiones enlazadas por CODIGOUNICO
PuestoTransfDistribucion -> UNIDADTRANSFDISTRIBUCION: 599 padres -> 1085 hijos (máx 3, 279 multi)
PuntoCarga -> CONEXIONCONSUMIDOR: 8126 padres -> 11400 hijos (máx 6, 1615 multi)
```

- El enlace por `CODIGOUNICO` debe ser cercano al 100 %.
- Los "multi" confirman que la jerarquía se leyó bien: **279 bancos** de varias
  unidades y **1.615 edificios** con varios medidores.

## Paso 2 — Ingesta del sistema comercial

```bash
lossan ingest-consumption dataset_cnel/comercial/consumo_historico.csv \
  --map cuenta_contrato=CUENTA_CONTRATO,year_month=PERIODO,kwh=CONSUMO_KWH,kvarh=CONSUMO_KVARH,estimated=LECTURA_ESTIMADA

lossan ingest-header dataset_cnel/comercial/cabecera_alimentador.csv \
  --map feeder_id=ALIMENTADOR,year_month=PERIODO,kwh=ENERGIA_KWH,kvarh=ENERGIA_KVARH
```

**Lo más importante que mirar** es la tasa de cruce comercial↔SIG:

```
"commercial_link": { "matched": 410400, "unmatched": 0, "match_rate": 1.0 }
```

Con datos reales, **por debajo del 95 % el sistema avisa**: un cruce bajo
invalida el balance porque hay consumo que no se asigna a ninguna conexión.

## Paso 3 — Ejecutar el análisis (todas las etapas)

```bash
lossan run
```

Recorre, por alimentador: topología y calidad → balance y cargabilidad → flujo
de potencia → estimación de estado → riesgo de hurto; y a nivel de sistema:
transferencias, reconciliación P/Q y plan de campaña.

Para una pasada rápida sin las fases pesadas (tamizaje masivo):

```bash
lossan run --level n1
```

## Paso 4 — Revisar los resultados

```bash
lossan status                    # avance y balance por alimentador
lossan feeder-report F0000       # elementos conectados + desglose de pérdidas
lossan dashboard                 # tablero web  ->  http://localhost:8501
```

---

## Qué revisar en cada etapa

| Etapa | Comando / salida | Qué debe verse |
|---|---|---|
| **Ingesta SIG** | `ingest-cnel-csv` | enlace `CODIGOUNICO` ~100 %; bancos y edificios multi detectados |
| **Ingesta comercial** | `ingest-consumption` | `match_rate` ≥ 0,95 |
| **Topología (F2)** | `feeder-report F0000` | nodos/tramos, clientes por traza, validación sin ciclos |
| **Calidad (R01–R25)** | dashboard → *Topología & Calidad* | reglas disparadas con valor sugerido |
| **Balance (F5)** | `lossan status` | `balance_closed = True`; PNT y técnicas en rango plausible |
| **Punto de carga (§5.3)** | GOLD `load_point_balance` | coincidencia intra-punto reduce la demanda ~35 % |
| **Flujo (F4)** | dashboard → *Flujo de potencia* | converge; V mín en pu razonable |
| **Estado (F6)** | dashboard → *Estado & Plan* | carga no contabilizada por zona |
| **Riesgo (F7)** | dashboard → *Riesgo de hurto* | ranking con razones SHAP |
| **Campaña (F8)** | dashboard → *Plan de campaña* | puestos seleccionados, ROI, Precision@k |

## Medir la precisión del modelo

```bash
python examples/calibrate.py --root data/lake --feeder F0000
```

Compara el ranking contra `verdad_terreno_hurtos.csv`: reporta **Precision@k**,
compara los métodos de PU learning y barre umbrales. Es el paso para decidir la
configuración antes de una campaña real.

## Publicar a GIS

```bash
lossan export-results --fmt fgdb --out salida/resultados
lossan report --feeder F0000                    # informe ejecutivo PDF
lossan inspection-sheet <ID_PUESTO>             # ficha para la cuadrilla
```

---

## Regenerar el dataset con otro tamaño

```bash
# 1) universo sintético (ajusta config/scale.yaml: feeders, clientes, meses)
lossan generate
# 2) exportarlo al formato CNEL
lossan generate-cnel --out mi_dataset
```

## Cuando llegue la GDB real

```bash
lossan cnel-domains "C:\ruta\SIGELEC.gdb" --domain "Phase"   # verificar dominios
lossan ingest-cnel  "C:\ruta\SIGELEC.gdb"                    # misma ruta, desde FGDB
```

Los pasos 2 a 4 son idénticos. Si algún dominio o nombre de campo difiere, se
ajusta `config/cnel_mapping.yaml` — sin tocar código.
