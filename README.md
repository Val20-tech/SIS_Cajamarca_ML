# SIS Cajamarca ML — Modelo Predictivo de Demanda Asistencial

> Modelo de Machine Learning (Random Forest) para predecir la cantidad mensual de atenciones del Seguro Integral de Salud (SIS) por establecimiento, servicio, sexo y grupo de edad en la región Cajamarca, Perú (2023–2024).

---

## Links del proyecto

| Recurso | URL |
|---|---|
| Dashboard interactivo (nube) | https://sis-cajamarca-ml.streamlit.app |
| Base de datos (Supabase) | https://supabase.com — proyecto: `hnwsjrnzrvvzqnrahhis` |
| Datos fuente (SIS) | https://www.datosabiertos.gob.pe |

---

## Resultados del modelo

| Métrica | Modelo completo (100 árboles) | Modelo lite (10 árboles) |
|---|---|---|
| R² | **0.6625** | 0.4086 |
| MAE | **4.96** atenciones | 5.94 atenciones |
| RMSE | **12.72** atenciones | 16.84 atenciones |
| Filas entrenamiento | 1,269,288 | 1,269,288 |
| Filas prueba | 317,322 | 317,322 |

### Variable más importante: `cod_servicio` (42.70%)

| Variable | Importancia |
|---|---|
| cod_servicio | 42.70% |
| ipress | 25.48% |
| grupo_edad | 9.55% |
| provincia | 7.95% |
| nivel_eess | 7.82% |
| mes | 3.66% |
| sexo | 2.13% |
| anio | 0.71% |

---

## Arquitectura del proyecto

El proyecto usa la **Arquitectura Medallion** (Bronze → Silver → Gold):

```
CSV originales (assets/)
        ↓ [07_carga_gold.py]
Bronze: auditoría en Supabase (bronze.meta_pipeline)
        ↓
Gold: datos agregados (gold.demanda_mensual_ipress) — 1,586,610 filas
        ↓ [08_modelo_random_forest.py]
modelo_rf.pkl (Random Forest entrenado)
        ↓ [08b_storage_y_predicciones.py]
Gold: predicciones (gold.ml_predicciones) — 317,322 filas
        ↓ [09_dashboard.py]
Dashboard Streamlit (localhost:8501 o streamlit.app)
```

---

## Estructura de archivos

```
SIS_Cajamarca_ML/
│
├── assets/                          ← CSV originales del SIS (NO en GitHub, ~20 GB)
│   ├── OPENDATA_DS_01_2023_01_06_ATENCIONES.csv
│   ├── OPENDATA_DS_01_2023_07_12_ATENCIONES.csv
│   ├── OPENDATA_DS_01_2024_01_06_ATENCIONES.csv
│   └── OPENDATA_DS_01_2024_07_12_ATENCIONES.csv
│
├── notebooks/                       ← Scripts Python del pipeline
│   ├── 01_ejecutar_esquemas.py      ← Crear esquemas en Supabase (via psycopg2)
│   ├── 02_silver_desde_csv.py       ← Exploración inicial de los CSV
│   ├── 03_transformar_silver.py     ← Bronze → Silver en Supabase
│   ├── 04_cargar_gold.py            ← Silver → Gold via RPC
│   ├── 05_bronze_a_gold_directo.py  ← Bronze → Gold directo (intento)
│   ├── 06_gold_por_partes.py        ← Bronze → Gold por chunks (intento)
│   ├── 07_carga_gold.py             ← ✅ SCRIPT PRINCIPAL: Bronze → Gold
│   ├── 08_modelo_random_forest.py   ← ✅ Entrenamiento del modelo RF
│   ├── 08b_storage_y_predicciones.py← ✅ Predicciones → Supabase
│   ├── 09_dashboard.py              ← ✅ Dashboard Streamlit
│   ├── 10_entrenar_modelo_ligero.py ← Modelo lite para Streamlit Cloud
│   ├── modelo_rf_lite.pkl           ← Modelo lite (14 MB, en GitHub)
│   └── modelo_rf.pkl                ← Modelo completo (1.5 GB, NO en GitHub)
│
├── sql/                             ← Scripts SQL para Supabase
│   ├── 00_setup_completo.sql        ← ✅ Setup inicial completo (ejecutar primero)
│   ├── bronze/
│   │   └── 01_crear_esquemas.sql    ← Esquemas + meta_pipeline
│   ├── silver/
│   │   └── 02_crear_silver_atenciones.sql
│   └── gold/
│       ├── 03_crear_gold_demanda.sql
│       └── 04_crear_ml_predicciones.sql
│
├── .env                             ← ⚠️ Credenciales Supabase (NO compartir)
├── .gitignore                       ← Archivos excluidos de GitHub
├── README.md                        ← Este archivo
└── requirements.txt                 ← Dependencias Python
```

---

## Requisitos previos

- Python 3.11 o superior
- Cuenta en [Supabase](https://supabase.com) (gratuita)
- Los 4 CSV del SIS descargados en la carpeta `assets/`

---

## Instalación paso a paso

### Paso 1 — Descargar el código
```bash
# Opción A: clonar con Git
git clone https://github.com/Val20-tech/SIS_Cajamarca_ML.git

# Opción B: descargar ZIP desde GitHub
# Clic en "Code" → "Download ZIP" → descomprimir
```

### Paso 2 — Instalar dependencias
```bash
pip install -r requirements.txt
```

### Paso 3 — Crear tu archivo `.env`
Crea un archivo llamado `.env` en la raíz del proyecto con tus credenciales de Supabase:
```
SUPABASE_URL=https://TU-PROYECTO.supabase.co
SUPABASE_KEY=TU-ANON-KEY
SUPABASE_DB_HOST=TU-HOST
SUPABASE_DB_USER=postgres.TU-PROYECTO
SUPABASE_DB_PASSWORD=TU-PASSWORD
SUPABASE_DB_NAME=postgres
SUPABASE_DB_PORT=6543
```

### Paso 4 — Crear la estructura en Supabase
1. Ir a **supabase.com** → tu proyecto → **SQL Editor**
2. Pegar el contenido de `sql/00_setup_completo.sql`
3. Clic en **Run**

### Paso 5 — Descargar los CSV del SIS
Ir a [datosabiertos.gob.pe](https://www.datosabiertos.gob.pe) y descargar:
- Atenciones SIS 2023 (semestre 1 y 2)
- Atenciones SIS 2024 (semestre 1 y 2)

Guardarlos en la carpeta `assets/`.

---

## Ejecución del pipeline

Ejecutar los scripts en este orden desde PowerShell o terminal:

```bash
# 1. Cargar datos a Gold en Supabase (~49 minutos)
python notebooks/07_carga_gold.py

# 2. Entrenar el modelo Random Forest (~33 minutos)
python notebooks/08_modelo_random_forest.py

# 3. Cargar predicciones a Supabase (~33 minutos)
python notebooks/08b_storage_y_predicciones.py

# 4. Lanzar el dashboard
python -m streamlit run notebooks/09_dashboard.py
```

Abrir el navegador en: **http://localhost:8501**

---

## El dashboard

El dashboard tiene 3 pestañas:

| Pestaña | Contenido |
|---|---|
| Métricas del Modelo | R², MAE, RMSE, distribución del target |
| Importancia de Variables | Gráfico de barras con contribución de cada variable |
| Predictor Interactivo | Selecciona provincia, IPRESS, mes, sexo, edad → predice atenciones |

**Versión online (modelo lite, R²=0.41):**
https://sis-cajamarca-ml.streamlit.app

**Versión local (modelo completo, R²=0.66):**
```bash
python -m streamlit run notebooks/09_dashboard.py
```

---

## Base de datos en Supabase

| Tabla | Esquema | Filas | Descripción |
|---|---|---|---|
| `meta_pipeline` | bronze | 1 | Log de auditoría de cargas |
| `demanda_mensual_ipress` | gold | 1,586,610 | Datos agregados para el modelo |
| `ml_predicciones` | gold | 317,322 | Predicciones del test set (20%) |

---

## El modelo Random Forest

### ¿Qué es Random Forest?
Es un conjunto (ensemble) de 100 árboles de decisión. Cada árbol aprende patrones diferentes de los datos. La predicción final es el promedio de los 100 árboles.

### Variables predictoras (features)
```
anio, mes, provincia, ipress, nivel_eess, cod_servicio, sexo, grupo_edad
```

### Variable objetivo (target)
```
total_atenciones — número de atenciones mensuales por combinación
```

### Parámetros del modelo
```python
RandomForestRegressor(
    n_estimators=100,     # 100 árboles
    min_samples_leaf=5,   # mínimo 5 muestras por hoja
    max_features='sqrt',  # sqrt(8) ≈ 3 variables por nodo
    n_jobs=-1,            # todos los núcleos CPU
    random_state=42       # reproducibilidad
)
```

### División train/test
- **80% entrenamiento:** 1,269,288 filas
- **20% prueba:** 317,322 filas
- **random_state=42** garantiza reproducibilidad

---

## Problemas conocidos y soluciones

| Problema | Causa | Solución |
|---|---|---|
| psycopg2 no conecta | Puerto PostgreSQL bloqueado por firewall | Usar API REST con `requests` |
| modelo_rf.pkl no cabe en GitHub | Pesa 1.5 GB (límite GitHub: 100 MB) | Excluido en `.gitignore`; usar modelo lite para nube |
| Encoding corrupto en CSV | CSV exportados con cp1252 en lugar de utf-8 | Normalizar nombres de columnas en script 07 |
| MemoryError al leer CSV | CSV de varios GB no caben en RAM | Usar `chunksize=100_000` en `pd.read_csv()` |
| Python 3.14 + plotly incompatible | Streamlit Cloud usa Python muy reciente | Archivo `.python-version` fuerza Python 3.11 |

---

## Dependencias principales

```
streamlit==1.57.0
plotly==5.18.0
scikit-learn==1.8.0
pandas==3.0.3
numpy==2.4.6
requests==2.34.2
python-dotenv==1.2.2
```

Ver lista completa en `requirements.txt`.

---

## Equipo

Proyecto desarrollado como parte del curso de Proyecto Productivo 2B — Instituto Continental, 2026.

| Integrante | Rol |
|---|---|
| ALARCON OLMEDO, VICTOR EFRAIN | (Modelado de Machine Learning) |
| ANASTACIO VALLEJOS, KEVIN GONZALO | (Ingeniería de datos y pipeline ETL) |
| COTERA HUAMAN, PILAR SANDY | (Arquitectura de base de datos y Supabase) |
| INUMA MACEDO, MARIANA MILAGROS | (Análisis estadístico y validación de métricas) |
| MEJÍA ROJAS, KEVIN OLLANTA | Desarrollo del dashboard y visualización) |
| MENDEZ MACEDO, KAREN YASELI | (Documentación y redacción del informe) |
| SALINAS LLANA, JHIM ANTHONY | (Control de calidad y gestión de GitHub) |

---

## Licencia

Proyecto académico — datos públicos del Ministerio de Salud del Perú (MINSA/SIS).
