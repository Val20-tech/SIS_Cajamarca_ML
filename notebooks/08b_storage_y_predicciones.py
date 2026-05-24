"""
08b_storage_y_predicciones.py
PREREQUISITOS (una sola vez en Supabase):
  1. SQL Editor → ejecutar sql/gold/04_crear_ml_predicciones.sql
  2. Dashboard > Storage > New bucket > nombre: modelos, tipo: Public

Luego ejecutar este script:
  python notebooks/08b_storage_y_predicciones.py

Pasos automaticos:
  [1] Carga modelo_rf.pkl
  [2] Sube pkl a Supabase Storage bucket 'modelos'
  [3] Descarga gold.demanda_mensual_ipress via REST (keyset pagination)
  [4] Recrea split 80/20 exacto y genera predicciones
  [5] Inserta predicciones en gold.ml_predicciones via REST
"""

import os, json, time, pickle
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

PAGE_DL      = 5_000    # filas por pagina en descarga (Supabase devuelve min(PAGE_DL, max-rows))
BATCH_INSERT = 500      # filas por POST en insercion
TEST_SIZE    = 0.2
RANDOM_STATE = 42
MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'modelo_rf.pkl')

# ── 1. Credenciales y modelo ─────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
URL = os.getenv('SUPABASE_URL')
KEY = os.getenv('SUPABASE_KEY')

if not URL or not KEY:
    raise SystemExit("ERROR: SUPABASE_URL o SUPABASE_KEY no encontrados en .env")

print("[1/5] Cargando modelo_rf.pkl...")
with open(MODEL_PATH, 'rb') as f:
    artifact = pickle.load(f)

model    = artifact['model']
encoders = artifact['encoders']
features = artifact['features']
cat_cols = artifact['cat_cols']
metrics  = artifact['metrics']
print(f"      R2={metrics['r2']}  MAE={metrics['mae']}  RMSE={metrics['rmse']}")

def hdrs_rest(schema='gold', returning='minimal'):
    return {
        'apikey':          KEY,
        'Authorization':   f'Bearer {KEY}',
        'Content-Type':    'application/json',
        'Prefer':          f'return={returning}',
        'Content-Profile': schema,
        'Accept-Profile':  schema,
    }

def hdrs_storage():
    return {'apikey': KEY, 'Authorization': f'Bearer {KEY}'}

# ── 2. Supabase Storage ──────────────────────────────────────────────────────
print("\n[2/5] Verificando modelo_rf.pkl para Storage...")
pkl_size_mb = os.path.getsize(MODEL_PATH) / 1_048_576
print(f"      Tamanio pkl : {pkl_size_mb:.0f} MB")
if pkl_size_mb > 50:
    print(f"      [SKIP] El archivo supera el limite de Supabase Storage (~50 MB).")
    print(f"      >> Alternativa: guarda el pkl en un servidor con mas capacidad")
    print(f"      >> El dashboard carga el modelo desde disco local (no requiere Storage)")
else:
    with open(MODEL_PATH, 'rb') as f:
        pkl_bytes = f.read()
    r = requests.post(
        f"{URL}/storage/v1/object/modelos/modelo_rf.pkl",
        headers={**hdrs_storage(), 'Content-Type': 'application/octet-stream',
                 'x-upsert': 'true'},
        data=pkl_bytes, timeout=60,
    )
    if r.status_code in (200, 201):
        print(f"      Subido: {pkl_size_mb:.1f} MB")
    else:
        print(f"      [WARN] Storage {r.status_code}: {r.text[:100]}")

# ── 3. Descargar gold.demanda_mensual_ipress (keyset) ───────────────────────
print("\n[3/5] Descargando gold.demanda_mensual_ipress via REST...")

# Total de filas
r = requests.get(
    f"{URL}/rest/v1/demanda_mensual_ipress",
    headers={**hdrs_rest(), 'Prefer': 'count=exact'},
    params={'select': 'id', 'limit': 1},
)
total = int(r.headers.get('Content-Range', '0-0/0').split('/')[-1])
print(f"      Total filas: {total:,}")

SELECT_COLS = 'id,' + ','.join(features + ['total_atenciones'])
dfs     = []
cursor  = 0
fetched = 0
t_dl    = time.time()

while True:
    r = requests.get(
        f"{URL}/rest/v1/demanda_mensual_ipress",
        headers=hdrs_rest(),
        params={
            'select': SELECT_COLS,
            'id':     f'gt.{cursor}',
            'order':  'id.asc',
            'limit':  PAGE_DL,
        },
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
print(f"\n      Shape descargado: {df.shape}  |  {time.time()-t_dl:.1f}s\n")

# ── 4. Codificar, split exacto y predecir ────────────────────────────────────
print("[4/5] Codificando, reproduciendo split 80/20 y prediciendo...")

for col in cat_cols:
    df[col] = df[col].fillna('DESCONOCIDO').astype(str).str.strip().str.upper()
    df[col] = encoders[col].transform(df[col])
for col in ['anio', 'mes']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
df['total_atenciones'] = pd.to_numeric(df['total_atenciones'], errors='coerce').fillna(0).astype(int)

X = df[features].values.astype(np.float32)
y = df['total_atenciones'].values.astype(np.float32)

# Split identico al entrenamiento (test_size=0.2, random_state=42)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"      Test set: {len(X_test):,} filas (exacto al entrenamiento)")

t_pred = time.time()
y_pred = model.predict(X_test)
print(f"      Prediccion: {len(y_pred):,} filas en {time.time()-t_pred:.1f}s")

# Decodificar etiquetas para el resultado final
df_res = pd.DataFrame(X_test, columns=features)
for col in cat_cols:
    df_res[col] = encoders[col].inverse_transform(df_res[col].astype(int))
for col in ['anio', 'mes']:
    df_res[col] = df_res[col].astype(int)

df_res['atenciones_real'] = y_test.astype(int)
df_res['atenciones_pred'] = np.round(y_pred, 2)
df_res['error_abs']       = np.abs(y_test - y_pred).round(2)

INSERT_COLS = [
    'anio', 'mes', 'provincia', 'ipress', 'nivel_eess',
    'cod_servicio', 'sexo', 'grupo_edad',
    'atenciones_real', 'atenciones_pred', 'error_abs',
]

# ── 5. Limpiar tabla e insertar predicciones via REST ────────────────────────
print(f"\n[5/5] Insertando {len(df_res):,} predicciones en gold.ml_predicciones...")

# Vaciar tabla antes de insertar
r = requests.delete(
    f"{URL}/rest/v1/ml_predicciones",
    headers=hdrs_rest(),
    params={'id': 'gte.0'},
)
print(f"      DELETE previo: HTTP {r.status_code}")

records   = df_res[INSERT_COLS].to_dict(orient='records')
n_batches = (len(records) + BATCH_INSERT - 1) // BATCH_INSERT
errores   = 0
t_ins     = time.time()

for i in range(n_batches):
    batch = records[i * BATCH_INSERT : (i + 1) * BATCH_INSERT]
    r = requests.post(
        f"{URL}/rest/v1/ml_predicciones",
        headers=hdrs_rest(),
        data=json.dumps(batch),
    )
    if r.status_code not in (200, 201):
        errores += 1
        print(f"      [ERROR] batch {i+1}: HTTP {r.status_code} {r.text[:100]}")

    if (i + 1) % 100 == 0 or (i + 1) == n_batches:
        done    = min((i + 1) * BATCH_INSERT, len(records))
        pct     = (i + 1) / n_batches * 100
        elapsed = time.time() - t_ins
        vel     = done / elapsed if elapsed > 0 else 0
        print(f"      Batch {i+1:>4}/{n_batches}  ({pct:5.1f}%)  "
              f"|  {done:>8,} filas  |  {elapsed:5.0f}s  |  {vel:,.0f} filas/s")

# ── Resumen ──────────────────────────────────────────────────────────────────
sep = "=" * 54
print(f"\n{sep}")
print(f"  {'COMPLETADO' if errores == 0 else 'COMPLETADO CON ERRORES'}")
print(f"  Storage : modelos/modelo_rf.pkl")
print(f"  Tabla   : gold.ml_predicciones — {len(records):,} filas")
print(f"  Errores : {errores} batches")
print(f"  Tiempo  : {time.time()-t_dl:.0f}s total")
print(sep)
