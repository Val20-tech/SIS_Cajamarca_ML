import os, glob, json, time
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv

BATCH_SIZE = 500

# ── 1. Credenciales ──────────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
URL = os.getenv('SUPABASE_URL')
KEY = os.getenv('SUPABASE_KEY')

def headers(schema, returning='minimal'):
    return {
        'apikey':          KEY,
        'Authorization':   f'Bearer {KEY}',
        'Content-Type':    'application/json',
        'Prefer':          f'return={returning}',
        'Content-Profile': schema,
        'Accept-Profile':  schema,
    }

print("[1/6] Credenciales cargadas")

# ── 2. Leer y filtrar los 4 CSV (Bronze) ─────────────────────────────────────
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

# ── 3. Limpiar en memoria (Silver efimero) ────────────────────────────────────
print("[3/6] Limpiando y transformando en memoria...")

KEEP = ['AÑO','MES','REGION','PROVINCIA','DISTRITO','IPRESS',
        'NIVEL_EESS','PLAN_SEGURO','COD_SERVICIO','DESC_SERVICIO',
        'SEXO','GRUPO_EDAD','ATENCIONES']

df = raw[KEEP].copy()

TEXT_COLS = ['REGION','PROVINCIA','DISTRITO','IPRESS','DESC_SERVICIO','SEXO','GRUPO_EDAD']
for col in TEXT_COLS:
    df[col] = df[col].str.strip().str.upper()

df['ATENCIONES'] = pd.to_numeric(df['ATENCIONES'], errors='coerce').fillna(0).astype(int)
df['AÑO'] = pd.to_numeric(df['AÑO'], errors='coerce').astype('Int16')
df['MES'] = pd.to_numeric(df['MES'], errors='coerce').astype('Int16')

df = df[df['ATENCIONES'] >= 0].dropna(subset=['AÑO','MES'])

# ── 4. Agregar: GROUP BY en pandas ───────────────────────────────────────────
print("[4/6] Agregando (GROUP BY) en pandas...")

GROUP_COLS = ['AÑO','MES','REGION','PROVINCIA','DISTRITO','IPRESS',
              'NIVEL_EESS','PLAN_SEGURO','COD_SERVICIO','DESC_SERVICIO',
              'SEXO','GRUPO_EDAD']

gold = (df.groupby(GROUP_COLS, as_index=False, dropna=False)
          .agg(total_atenciones=('ATENCIONES', 'sum')))

gold.rename(columns={
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
}, inplace=True)

gold['anio'] = gold['anio'].astype(int)
gold['mes']  = gold['mes'].astype(int)
gold['total_atenciones'] = gold['total_atenciones'].astype(int)

total_filas = len(gold)
print(f"      Filas Gold tras agregacion: {total_filas:,}")
print(f"      Factor de reduccion: {len(raw)/total_filas:.1f}x")

# ── 5. Limpiar tabla Gold existente ──────────────────────────────────────────
print("[5/6] Limpiando gold.demanda_mensual_ipress antes de cargar...")
r = requests.delete(
    f"{URL}/rest/v1/demanda_mensual_ipress",
    headers={**headers('gold'), 'Prefer': 'return=minimal'},
    params={'id': 'gte.0'}
)
print(f"      DELETE status: {r.status_code}")

# ── 5b. Registrar inicio en meta_pipeline ────────────────────────────────────
meta_inicio = {
    'tabla':            'gold.demanda_mensual_ipress',
    'tipo_carga':       'bronze_a_gold_directo',
    'fecha_inicio':     datetime.now().isoformat(),
    'filas_procesadas': total_filas,
    'estado':           'en_proceso',
}
r = requests.post(
    f"{URL}/rest/v1/meta_pipeline",
    headers=headers('bronze', returning='representation'),
    data=json.dumps(meta_inicio)
)
meta_id = r.json()[0]['id'] if r.status_code in (200, 201) else None
print(f"      meta_pipeline id={meta_id}")

# ── 6. Cargar Gold en batches ─────────────────────────────────────────────────
print(f"[6/6] Cargando {total_filas:,} filas a Gold en batches de {BATCH_SIZE}...")

records   = gold.to_dict(orient='records')
n_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
errores   = 0
t0        = time.time()

for i in range(n_batches):
    batch = records[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    r = requests.post(
        f"{URL}/rest/v1/demanda_mensual_ipress",
        headers=headers('gold'),
        data=json.dumps(batch)
    )
    if r.status_code not in (200, 201):
        errores += 1
        print(f"      [ERROR] batch {i+1}: {r.status_code} {r.text[:120]}")

    if (i + 1) % 100 == 0 or (i + 1) == n_batches:
        pct     = (i + 1) / n_batches * 100
        elapsed = time.time() - t0
        filas_ok = min((i + 1) * BATCH_SIZE, total_filas)
        print(f"      Batch {i+1:>5}/{n_batches} ({pct:5.1f}%) | "
              f"{filas_ok:>8,} filas | {elapsed:6.1f}s")

# ── Actualizar meta_pipeline ──────────────────────────────────────────────────
estado = 'completado' if errores == 0 else f'errores: {errores} batches'
if meta_id:
    requests.patch(
        f"{URL}/rest/v1/meta_pipeline?id=eq.{meta_id}",
        headers={**headers('bronze'), 'Prefer': 'return=minimal'},
        data=json.dumps({
            'fecha_fin':        datetime.now().isoformat(),
            'estado':           estado,
            'observaciones':    f"{total_filas:,} filas gold en {n_batches} batches",
        })
    )

elapsed_total = time.time() - t0
print(f"\n{'='*55}")
print(f"  GOLD CARGADO")
print(f"  Filas cargadas : {total_filas:,}")
print(f"  Errores        : {errores}")
print(f"  Tiempo total   : {elapsed_total:.1f}s")
print(f"  Estado         : {estado}")
print(f"{'='*55}")
