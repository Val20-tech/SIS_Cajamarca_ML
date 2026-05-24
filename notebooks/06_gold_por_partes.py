import os, glob, json, time
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv

CHUNK_SIZE  = 100_000
BATCH_SIZE  = 500

GROUP_COLS = ['AÑO','MES','REGION','PROVINCIA','DISTRITO','IPRESS',
              'NIVEL_EESS','PLAN_SEGURO','COD_SERVICIO','DESC_SERVICIO',
              'SEXO','GRUPO_EDAD']

TEXT_COLS  = ['REGION','PROVINCIA','DISTRITO','IPRESS','DESC_SERVICIO','SEXO','GRUPO_EDAD']

RENAME_MAP = {
    'AÑO':          'anio',   'MES':          'mes',
    'REGION':       'region', 'PROVINCIA':    'provincia',
    'DISTRITO':     'distrito','IPRESS':       'ipress',
    'NIVEL_EESS':   'nivel_eess','PLAN_SEGURO':'plan_seguro',
    'COD_SERVICIO': 'cod_servicio','DESC_SERVICIO':'desc_servicio',
    'SEXO':         'sexo',   'GRUPO_EDAD':   'grupo_edad',
}

# ── 1. Credenciales ──────────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
URL = os.getenv('SUPABASE_URL')
KEY = os.getenv('SUPABASE_KEY')

def hdrs(schema, returning='minimal'):
    return {
        'apikey':          KEY,
        'Authorization':   f'Bearer {KEY}',
        'Content-Type':    'application/json',
        'Prefer':          f'return={returning}',
        'Content-Profile': schema,
        'Accept-Profile':  schema,
    }

print("[1/5] Credenciales cargadas")

# ── 2. Procesar cada CSV por chunks ──────────────────────────────────────────
base      = os.path.join(os.path.dirname(__file__), '..', 'assets')
csv_files = sorted(glob.glob(os.path.join(base, '*.csv')))
print(f"[2/5] Procesando {len(csv_files)} CSV en chunks de {CHUNK_SIZE:,} filas...\n")

parciales = []

for csv_idx, path in enumerate(csv_files, 1):
    nombre     = os.path.basename(path)
    acum       = []          # chunks filtrados+agrupados de este CSV
    filas_raw  = 0
    filas_caj  = 0
    n_chunks   = 0
    t_csv      = time.time()

    print(f"  [{csv_idx}/4] {nombre}")

    for chunk in pd.read_csv(path, encoding='utf-8', sep=',',
                             dtype=str, chunksize=CHUNK_SIZE):
        n_chunks  += 1
        filas_raw += len(chunk)

        # Filtrar Cajamarca
        mask  = chunk['REGION'].str.strip().str.upper() == 'CAJAMARCA'
        chunk = chunk[mask].copy()
        if chunk.empty:
            continue
        filas_caj += len(chunk)

        # Limpiar texto
        for col in TEXT_COLS:
            chunk[col] = chunk[col].str.strip().str.upper()

        # Tipos numericos
        chunk['ATENCIONES'] = pd.to_numeric(chunk['ATENCIONES'], errors='coerce').fillna(0).astype(int)
        chunk['AÑO'] = pd.to_numeric(chunk['AÑO'], errors='coerce').astype('Int16')
        chunk['MES'] = pd.to_numeric(chunk['MES'], errors='coerce').astype('Int16')

        chunk = chunk[chunk['ATENCIONES'] >= 0].dropna(subset=['AÑO','MES'])

        # Agrupar chunk y acumular
        grp = (chunk.groupby(GROUP_COLS, as_index=False, dropna=False)
                    .agg(ATENCIONES=('ATENCIONES','sum')))
        acum.append(grp)

        if n_chunks % 10 == 0:
            print(f"        chunk {n_chunks:>3} | {filas_raw:>10,} leidas | "
                  f"{filas_caj:>8,} Cajamarca")

    # Reagrupar acumulado del CSV
    parcial = (pd.concat(acum, ignore_index=True)
                 .groupby(GROUP_COLS, as_index=False, dropna=False)
                 .agg(ATENCIONES=('ATENCIONES','sum')))
    parciales.append(parcial)

    print(f"        Filas totales: {filas_raw:,} | Cajamarca: {filas_caj:,} | "
          f"Grupos parciales: {len(parcial):,} | {time.time()-t_csv:.1f}s\n")

# ── 3. Combinar los 4 parciales y reagrupar ───────────────────────────────────
print("[3/5] Combinando 4 parciales y reagrupando...")
gold = (pd.concat(parciales, ignore_index=True)
          .groupby(GROUP_COLS, as_index=False, dropna=False)
          .agg(total_atenciones=('ATENCIONES','sum')))

gold.rename(columns=RENAME_MAP, inplace=True)
gold['anio'] = gold['anio'].astype(int)
gold['mes']  = gold['mes'].astype(int)
gold['total_atenciones'] = gold['total_atenciones'].astype(int)

total_filas = len(gold)
print(f"      Filas Gold finales: {total_filas:,}")

# ── 4. Limpiar Gold + registrar inicio ───────────────────────────────────────
print("[4/5] Limpiando tabla Gold y registrando en meta_pipeline...")

r = requests.delete(
    f"{URL}/rest/v1/demanda_mensual_ipress",
    headers=hdrs('gold'), params={'id': 'gte.0'}
)
print(f"      DELETE gold: HTTP {r.status_code}")

meta = {
    'tabla':            'gold.demanda_mensual_ipress',
    'tipo_carga':       'bronze_gold_por_partes',
    'fecha_inicio':     datetime.now().isoformat(),
    'filas_procesadas': total_filas,
    'estado':           'en_proceso',
}
r = requests.post(f"{URL}/rest/v1/meta_pipeline",
                  headers=hdrs('bronze', returning='representation'),
                  data=json.dumps(meta))
meta_id = r.json()[0]['id'] if r.status_code in (200, 201) else None
print(f"      meta_pipeline id={meta_id}")

# ── 5. Cargar Gold en batches ─────────────────────────────────────────────────
print(f"[5/5] Cargando {total_filas:,} filas en batches de {BATCH_SIZE}...")

records   = gold.to_dict(orient='records')
n_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
errores   = 0
t0        = time.time()

for i in range(n_batches):
    batch = records[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    r = requests.post(
        f"{URL}/rest/v1/demanda_mensual_ipress",
        headers=hdrs('gold'),
        data=json.dumps(batch)
    )
    if r.status_code not in (200, 201):
        errores += 1
        print(f"      [ERROR] batch {i+1}: {r.status_code} {r.text[:100]}")

    if (i + 1) % 100 == 0 or (i + 1) == n_batches:
        pct     = (i + 1) / n_batches * 100
        elapsed = time.time() - t0
        filas_ok = min((i + 1) * BATCH_SIZE, total_filas)
        print(f"      Batch {i+1:>5}/{n_batches} ({pct:5.1f}%) | "
              f"{filas_ok:>8,} filas | {elapsed:5.1f}s")

# Actualizar meta_pipeline
estado = 'completado' if errores == 0 else f'errores: {errores} batches'
if meta_id:
    requests.patch(
        f"{URL}/rest/v1/meta_pipeline?id=eq.{meta_id}",
        headers={**hdrs('bronze'), 'Prefer': 'return=minimal'},
        data=json.dumps({
            'fecha_fin':     datetime.now().isoformat(),
            'estado':        estado,
            'observaciones': f"{total_filas:,} filas gold en {n_batches} batches | errores: {errores}",
        })
    )

elapsed_total = time.time() - t0
print(f"\n{'='*55}")
print(f"  GOLD CARGADO EXITOSAMENTE")
print(f"  Filas cargadas : {total_filas:,}")
print(f"  Errores        : {errores}")
print(f"  Tiempo carga   : {elapsed_total:.1f}s")
print(f"  Estado         : {estado}")
print(f"{'='*55}")
