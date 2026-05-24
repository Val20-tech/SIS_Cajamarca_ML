import os, glob, time, json
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv

BATCH_SIZE = 2000

# ── 1. Credenciales ──────────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
URL = os.getenv('SUPABASE_URL')
KEY = os.getenv('SUPABASE_KEY')

def _headers(schema, returning='minimal'):
    return {
        'apikey':           KEY,
        'Authorization':    f'Bearer {KEY}',
        'Content-Type':     'application/json',
        'Prefer':           f'return={returning}',
        'Content-Profile':  schema,
        'Accept-Profile':   schema,
    }

print("[1/6] Credenciales cargadas")

# ── 2. Leer y filtrar los 4 CSV ───────────────────────────────────────────────
base = os.path.join(os.path.dirname(__file__), '..', 'assets')
csv_files = sorted(glob.glob(os.path.join(base, '*.csv')))
print(f"[2/6] Leyendo {len(csv_files)} archivos CSV...")

frames = []
for path in csv_files:
    df = pd.read_csv(path, encoding='utf-8', sep=',', dtype=str)
    df = df[df['REGION'].str.strip().str.upper() == 'CAJAMARCA']
    frames.append(df)
    print(f"      {os.path.basename(path)}: {len(df):,} filas Cajamarca")

raw = pd.concat(frames, ignore_index=True)
print(f"      Total concatenado: {len(raw):,} filas")

# ── 3. Transformar ────────────────────────────────────────────────────────────
print("[3/6] Transformando...")

KEEP = ['AÑO','MES','REGION','PROVINCIA','DISTRITO','IPRESS',
        'NIVEL_EESS','PLAN_SEGURO','COD_SERVICIO','DESC_SERVICIO',
        'SEXO','GRUPO_EDAD','ATENCIONES']

df = raw[KEEP].copy()

# Texto en mayúsculas
TEXT_COLS = ['REGION','PROVINCIA','DISTRITO','IPRESS','DESC_SERVICIO','SEXO','GRUPO_EDAD']
for col in TEXT_COLS:
    df[col] = df[col].str.strip().str.upper()

# Tipos numéricos
df['ATENCIONES'] = pd.to_numeric(df['ATENCIONES'], errors='coerce').fillna(0).astype(int)
df['AÑO'] = pd.to_numeric(df['AÑO'], errors='coerce').astype('Int16')
df['MES'] = pd.to_numeric(df['MES'], errors='coerce').astype('Int16')

# Filtrar negativos/nulos en ATENCIONES
antes = len(df)
df = df[df['ATENCIONES'] >= 0].dropna(subset=['AÑO','MES'])
print(f"      Filas eliminadas (negativas/nulas): {antes - len(df):,}")

# Deduplicar
antes = len(df)
df.drop_duplicates(inplace=True)
print(f"      Duplicados eliminados: {antes - len(df):,}")

# Renombrar para la BD (sin caracteres especiales)
df.rename(columns={
    'AÑO':          'anio',
    'MES':          'mes',
    'REGION':       'region',
    'PROVINCIA':    'provincia',
    'DISTRITO':     'distrito',
    'IPRESS':       'ipress',
    'NIVEL_EESS':   'nivel_eess',
    'PLAN_SEGURO':  'plan_seguro',
    'COD_SERVICIO': 'cod_servicio',
    'DESC_SERVICIO':'desc_servicio',
    'SEXO':         'sexo',
    'GRUPO_EDAD':   'grupo_edad',
    'ATENCIONES':   'atenciones',
}, inplace=True)

# Convertir Int16 a int nativo para JSON
df['anio'] = df['anio'].astype(int)
df['mes']  = df['mes'].astype(int)

total_filas = len(df)
print(f"      Filas listas para carga: {total_filas:,}")

# ── 4. Registrar inicio en meta_pipeline ─────────────────────────────────────
print("[4/6] Registrando inicio en bronze.meta_pipeline...")

meta_inicio = {
    'tabla':            'silver.atenciones_cajamarca',
    'tipo_carga':       'carga_inicial',
    'fecha_inicio':     datetime.now().isoformat(),
    'filas_procesadas': total_filas,
    'estado':           'en_proceso',
}
r = requests.post(
    f"{URL}/rest/v1/meta_pipeline",
    headers=_headers('bronze', returning='representation'),
    data=json.dumps(meta_inicio)
)
if r.status_code not in (200, 201):
    print(f"      [AVISO] meta_pipeline inicio: {r.status_code} {r.text[:200]}")
    meta_id = None
else:
    meta_id = r.json()[0]['id']
    print(f"      Registro meta_pipeline id={meta_id}")

# ── 5. Insertar en batches via REST API ───────────────────────────────────────
print(f"[5/6] Cargando a silver.atenciones_cajamarca en batches de {BATCH_SIZE:,}...")

records  = df.to_dict(orient='records')
n_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
errores   = 0
t0        = time.time()

for i in range(n_batches):
    batch = records[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    r = requests.post(
        f"{URL}/rest/v1/atenciones_cajamarca",
        headers=_headers('silver'),
        data=json.dumps(batch)
    )
    if r.status_code not in (200, 201):
        errores += 1
        print(f"      [ERROR] batch {i+1}/{n_batches}: {r.status_code} {r.text[:150]}")

    if (i + 1) % 50 == 0 or (i + 1) == n_batches:
        pct      = (i + 1) / n_batches * 100
        elapsed  = time.time() - t0
        filas_ok = min((i + 1) * BATCH_SIZE, total_filas)
        print(f"      Batch {i+1:>4}/{n_batches} ({pct:5.1f}%) | "
              f"{filas_ok:>9,} filas | {elapsed:6.1f}s")

# ── 6. Actualizar meta_pipeline ───────────────────────────────────────────────
estado_final = 'completado' if errores == 0 else f'errores: {errores} batches'
print(f"[6/6] Actualizando meta_pipeline -> estado: {estado_final}")

if meta_id is not None:
    meta_fin = {
        'fecha_fin':        datetime.now().isoformat(),
        'estado':           estado_final,
        'observaciones':    f"{total_filas:,} filas cargadas en {n_batches} batches",
    }
    r = requests.patch(
        f"{URL}/rest/v1/meta_pipeline?id=eq.{meta_id}",
        headers=_headers('bronze'),
        data=json.dumps(meta_fin)
    )
    if r.status_code not in (200, 201, 204):
        print(f"      [AVISO] meta_pipeline update: {r.status_code} {r.text[:200]}")

elapsed_total = time.time() - t0
print(f"\n{'='*55}")
print(f"  CARGA COMPLETADA")
print(f"  Filas cargadas : {total_filas:,}")
print(f"  Batches        : {n_batches} de {BATCH_SIZE:,} filas")
print(f"  Errores        : {errores}")
print(f"  Tiempo total   : {elapsed_total:.1f}s")
print(f"{'='*55}")
