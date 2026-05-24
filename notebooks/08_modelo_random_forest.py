"""
08_modelo_random_forest.py
Lee gold.demanda_mensual_ipress desde Supabase via API REST,
entrena Random Forest Regressor para demanda asistencial SIS-Cajamarca.
"""

import os, time, pickle
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Configuración ────────────────────────────────────────────────────────────
PAGE_SIZE        = 1_000
TARGET           = 'total_atenciones'
CAT_COLS         = ['provincia', 'ipress', 'nivel_eess', 'cod_servicio', 'sexo', 'grupo_edad']
NUM_COLS         = ['anio', 'mes']
FEATURES         = NUM_COLS + CAT_COLS
# Incluimos 'id' para keyset pagination (se descarta antes del modelo)
SELECT_COLS      = 'id,' + ','.join(FEATURES + [TARGET])

N_ESTIMATORS     = 100
MIN_SAMPLES_LEAF = 5
TEST_SIZE        = 0.2
RANDOM_STATE     = 42

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelo_rf.pkl')

# ── 1. Credenciales ──────────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
URL = os.getenv('SUPABASE_URL')
KEY = os.getenv('SUPABASE_KEY')

if not URL or not KEY:
    raise SystemExit("ERROR: SUPABASE_URL o SUPABASE_KEY no encontrados en .env")

def hdrs(extra=None):
    h = {
        'apikey':        KEY,
        'Authorization': f'Bearer {KEY}',
        'Accept-Profile':'gold',
    }
    if extra:
        h.update(extra)
    return h

print(f"[1/6] Credenciales cargadas — {URL}\n")

# ── 2. Obtener total de filas ────────────────────────────────────────────────
print("[2/6] Consultando total de filas en gold.demanda_mensual_ipress...")
r = requests.get(
    f"{URL}/rest/v1/demanda_mensual_ipress",
    headers=hdrs({'Prefer': 'count=exact'}),
    params={'select': 'id', 'limit': 1},
)
content_range = r.headers.get('Content-Range', '0-0/0')
total   = int(content_range.split('/')[-1])
n_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
print(f"      Total filas : {total:,}")
print(f"      Paginas     : {n_pages:,} x {PAGE_SIZE:,} filas\n")

# ── 3. Descargar todas las filas — keyset pagination (id > cursor) ───────────
# Evita el timeout de OFFSET alto usando WHERE id > last_id sobre índice B-tree
print("[3/6] Descargando datos via REST API (keyset pagination)...")
dfs      = []
cursor   = 0          # last id seen; empieza en 0 para incluir id=1
fetched  = 0
t_dl     = time.time()

while True:
    r = requests.get(
        f"{URL}/rest/v1/demanda_mensual_ipress",
        headers=hdrs(),
        params={
            'select': SELECT_COLS,
            'id':     f'gt.{cursor}',
            'order':  'id.asc',
            'limit':  PAGE_SIZE,
        },
    )
    if r.status_code != 200:
        raise SystemExit(f"ERROR REST {r.status_code}: {r.text[:200]}")

    batch = r.json()
    if not batch:
        break

    df_batch = pd.DataFrame(batch)
    cursor   = int(df_batch['id'].iloc[-1])
    dfs.append(df_batch)
    fetched += len(batch)

    if fetched % 50_000 == 0 or fetched >= total:
        elapsed = time.time() - t_dl
        vel     = fetched / elapsed if elapsed > 0 else 0
        pct     = fetched / total * 100
        print(f"      {fetched:>10,} / {total:,}  ({pct:5.1f}%)  |  "
              f"{elapsed:6.1f}s  |  {vel:,.0f} filas/s  |  cursor={cursor:,}")

df = pd.concat(dfs, ignore_index=True)
del dfs
df.drop(columns=['id'], inplace=True)   # id solo se usó como cursor
print(f"\n      Shape: {df.shape}  |  Tiempo descarga: {time.time()-t_dl:.1f}s\n")

# ── 4. Preparar features ─────────────────────────────────────────────────────
print("[4/6] Preparando features y codificando variables categoricas...")

df['anio']  = pd.to_numeric(df['anio'],  errors='coerce')
df['mes']   = pd.to_numeric(df['mes'],   errors='coerce')
df[TARGET]  = pd.to_numeric(df[TARGET],  errors='coerce')

for col in CAT_COLS:
    df[col] = df[col].fillna('DESCONOCIDO').astype(str).str.strip().str.upper()

df.dropna(subset=['anio', 'mes', TARGET], inplace=True)
df['anio']  = df['anio'].astype(int)
df['mes']   = df['mes'].astype(int)
df[TARGET]  = df[TARGET].astype(int)

print(f"      Filas tras limpieza : {len(df):,}")
print()

encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"      {col:20s}: {len(le.classes_):>5,} categorias")

X = df[FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32)

print(f"\n      X shape : {X.shape}")
print(f"      y stats : min={y.min():.0f}  max={y.max():.0f}  "
      f"media={y.mean():.2f}  mediana={np.median(y):.0f}\n")

# ── 5. Train / Test split ────────────────────────────────────────────────────
print("[5/6] Dividiendo train/test 80/20 y entrenando Random Forest...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"      Train : {len(X_train):,} filas")
print(f"      Test  : {len(X_test):,}  filas")
print(f"      RF    : {N_ESTIMATORS} arboles | min_samples_leaf={MIN_SAMPLES_LEAF} | n_jobs=-1\n")

t_rf = time.time()
rf = RandomForestRegressor(
    n_estimators     = N_ESTIMATORS,
    min_samples_leaf = MIN_SAMPLES_LEAF,
    max_features     = 'sqrt',
    n_jobs           = -1,
    random_state     = RANDOM_STATE,
    verbose          = 1,
)
rf.fit(X_train, y_train)
t_rf_fin = time.time() - t_rf
print(f"\n      Entrenamiento completado en {t_rf_fin:.1f}s\n")

# ── 6. Evaluar y guardar ─────────────────────────────────────────────────────
print("[6/6] Evaluando modelo y guardando artefactos...")
y_pred = rf.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2     = r2_score(y_test, y_pred)

print(f"\n  {'='*40}")
print(f"  METRICAS (test {TEST_SIZE*100:.0f}%)")
print(f"  {'='*40}")
print(f"  MAE   : {mae:>10.4f}  atenciones")
print(f"  RMSE  : {rmse:>10.4f}  atenciones")
print(f"  R2    : {r2:>10.4f}")
print(f"  {'='*40}\n")

# Feature importances top-10
importances = pd.Series(rf.feature_importances_, index=FEATURES)
top10 = importances.sort_values(ascending=False).head(10)

print("  Top 10 variables mas importantes:")
print("  " + "-" * 44)
for rank, (feat, imp) in enumerate(top10.items(), 1):
    bar = "#" * int(imp * 60)
    print(f"  {rank:>2}. {feat:20s}  {imp:.4f}  {bar}")

# Guardar modelo + encoders + metadata
artifact = {
    'model':    rf,
    'encoders': encoders,
    'features': FEATURES,
    'target':   TARGET,
    'cat_cols': CAT_COLS,
    'num_cols': NUM_COLS,
    'metrics':  {
        'mae':     round(float(mae), 4),
        'rmse':    round(float(rmse), 4),
        'r2':      round(float(r2), 4),
        'n_train': int(len(X_train)),
        'n_test':  int(len(X_test)),
    },
}
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(artifact, f)

print(f"\n  Modelo guardado : {MODEL_PATH}")
print(f"  Tiempo total    : {time.time()-t_dl:.1f}s (descarga + entreno)")
print(f"  Estado          : COMPLETADO")
