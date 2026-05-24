"""
10_entrenar_modelo_ligero.py
Entrena Random Forest LIGERO para Streamlit Cloud (< 100 MB)

Diferencias vs 08_modelo_random_forest.py:
  - n_estimators  : 100  → 10
  - min_samples_leaf:  5  → 50   (arboles mas cortos = modelo mas pequeno)
  - Salida        : notebooks/modelo_rf_lite.pkl

Descarga gold.demanda_mensual_ipress desde Supabase via REST (keyset pagination).
"""

import os, json, time, pickle
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Hiperparametros LITE ─────────────────────────────────────────────────────
N_ESTIMATORS     = 10
MIN_SAMPLES_LEAF = 50       # mayor que el full (5) -> arboles mas cortos
MAX_FEATURES     = 'sqrt'
TEST_SIZE        = 0.2
RANDOM_STATE     = 42
PAGE_SIZE        = 1_000

TARGET   = 'total_atenciones'
CAT_COLS = ['provincia', 'ipress', 'nivel_eess', 'cod_servicio', 'sexo', 'grupo_edad']
NUM_COLS = ['anio', 'mes']
FEATURES = NUM_COLS + CAT_COLS

LITE_PATH = os.path.join(os.path.dirname(__file__), 'modelo_rf_lite.pkl')

# ── 1. Credenciales ──────────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
URL = os.getenv('SUPABASE_URL')
KEY = os.getenv('SUPABASE_KEY')

if not URL or not KEY:
    raise SystemExit("ERROR: SUPABASE_URL o SUPABASE_KEY no encontrados en .env")

def hdrs():
    return {
        'apikey':        KEY,
        'Authorization': f'Bearer {KEY}',
        'Accept-Profile':'gold',
    }

print(f"[1/6] Credenciales cargadas — {URL}")

# ── 2. Total de filas ────────────────────────────────────────────────────────
print("[2/6] Consultando total de filas...")
r = requests.get(
    f"{URL}/rest/v1/demanda_mensual_ipress",
    headers={**hdrs(), 'Prefer': 'count=exact'},
    params={'select': 'id', 'limit': 1},
)
total = int(r.headers.get('Content-Range', '0-0/0').split('/')[-1])
print(f"      Total filas : {total:,}\n")

# ── 3. Descarga keyset pagination ────────────────────────────────────────────
SELECT_COLS = 'id,' + ','.join(FEATURES + [TARGET])
print("[3/6] Descargando datos (keyset pagination)...")

dfs, cursor, fetched = [], 0, 0
t_dl = time.time()

while True:
    r = requests.get(
        f"{URL}/rest/v1/demanda_mensual_ipress",
        headers=hdrs(),
        params={'select': SELECT_COLS, 'id': f'gt.{cursor}',
                'order': 'id.asc', 'limit': PAGE_SIZE},
    )
    if r.status_code != 200:
        raise SystemExit(f"ERROR REST {r.status_code}: {r.text[:200]}")
    batch = r.json()
    if not batch:
        break

    df_b    = pd.DataFrame(batch)
    cursor  = int(df_b['id'].iloc[-1])
    fetched += len(batch)
    dfs.append(df_b.drop(columns=['id']))

    if fetched % 100_000 == 0 or fetched >= total:
        elapsed = time.time() - t_dl
        vel     = fetched / elapsed if elapsed > 0 else 0
        pct     = fetched / total * 100
        print(f"      {fetched:>10,} / {total:,}  ({pct:5.1f}%)  "
              f"|  {elapsed:6.0f}s  |  {vel:,.0f} filas/s")

df = pd.concat(dfs, ignore_index=True)
del dfs
print(f"\n      Shape: {df.shape}  |  Tiempo descarga: {time.time()-t_dl:.1f}s\n")

# ── 4. Preparar features ─────────────────────────────────────────────────────
print("[4/6] Codificando variables categoricas...")

df['anio']  = pd.to_numeric(df['anio'],  errors='coerce')
df['mes']   = pd.to_numeric(df['mes'],   errors='coerce')
df[TARGET]  = pd.to_numeric(df[TARGET],  errors='coerce')

for col in CAT_COLS:
    df[col] = df[col].fillna('DESCONOCIDO').astype(str).str.strip().str.upper()

df.dropna(subset=['anio', 'mes', TARGET], inplace=True)
df['anio']  = df['anio'].astype(int)
df['mes']   = df['mes'].astype(int)
df[TARGET]  = df[TARGET].astype(int)

encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"      {col:20s}: {len(le.classes_):>5,} categorias")

X = df[FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32)
print(f"\n      X shape : {X.shape}")
print(f"      y stats : min={y.min():.0f}  max={y.max():.0f}  media={y.mean():.2f}\n")

# ── 5. Train / Test split + entrenar RF LITE ─────────────────────────────────
print("[5/6] Dividiendo train/test 80/20 y entrenando RF LITE...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"      Train : {len(X_train):,} filas")
print(f"      Test  : {len(X_test):,} filas")
print(f"      RF    : {N_ESTIMATORS} arboles | "
      f"min_samples_leaf={MIN_SAMPLES_LEAF} | n_jobs=-1\n")

t_rf = time.time()
rf = RandomForestRegressor(
    n_estimators     = N_ESTIMATORS,
    min_samples_leaf = MIN_SAMPLES_LEAF,
    max_features     = MAX_FEATURES,
    n_jobs           = -1,
    random_state     = RANDOM_STATE,
    verbose          = 1,
)
rf.fit(X_train, y_train)
print(f"\n      Entrenamiento completado en {time.time()-t_rf:.1f}s\n")

# ── 6. Evaluar y guardar ─────────────────────────────────────────────────────
print("[6/6] Evaluando y guardando modelo LITE...")
y_pred = rf.predict(X_test)
mae    = float(mean_absolute_error(y_test, y_pred))
rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2     = float(r2_score(y_test, y_pred))

print(f"\n  {'='*42}")
print(f"  METRICAS LITE (test {TEST_SIZE*100:.0f}%)")
print(f"  {'='*42}")
print(f"  MAE   : {mae:>10.4f}  atenciones")
print(f"  RMSE  : {rmse:>10.4f}  atenciones")
print(f"  R2    : {r2:>10.4f}")
print(f"  {'='*42}\n")

# Feature importances
importances = pd.Series(rf.feature_importances_, index=FEATURES)
print("  Top variables:")
for feat, imp in importances.sort_values(ascending=False).items():
    bar = "#" * int(imp * 40)
    print(f"  {feat:20s}  {imp:.4f}  {bar}")

# Guardar artefacto
artifact = {
    'model':    rf,
    'encoders': encoders,
    'features': FEATURES,
    'target':   TARGET,
    'cat_cols': CAT_COLS,
    'num_cols': NUM_COLS,
    'metrics':  {
        'mae':     round(mae, 4),
        'rmse':    round(rmse, 4),
        'r2':      round(r2, 4),
        'n_train': int(len(X_train)),
        'n_test':  int(len(X_test)),
    },
    'model_type': 'lite',
    'n_estimators': N_ESTIMATORS,
    'min_samples_leaf': MIN_SAMPLES_LEAF,
}
with open(LITE_PATH, 'wb') as f:
    pickle.dump(artifact, f)

size_mb = os.path.getsize(LITE_PATH) / 1_048_576
status  = "OK - apto para GitHub/Streamlit Cloud" if size_mb < 100 else "ADVERTENCIA - supera 100 MB"

print(f"\n  Modelo guardado : {LITE_PATH}")
print(f"  Tamanio         : {size_mb:.1f} MB  [{status}]")
print(f"  Tiempo total    : {time.time()-t_dl:.1f}s")
