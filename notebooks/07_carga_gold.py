"""
Bronze → Gold directo (en memoria)
GROUP BY: anio, mes, provincia, ipress, nivel_eess, cod_servicio, sexo, grupo_edad
Carga a gold.demanda_mensual_ipress via REST API en batches de 500
"""

import os, glob, json, time
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv

CHUNK_SIZE = 100_000
BATCH_SIZE = 500

GROUP_COLS = [
    'AÑO', 'MES', 'PROVINCIA', 'IPRESS',
    'NIVEL_EESS', 'COD_SERVICIO', 'SEXO', 'GRUPO_EDAD',
]

TEXT_COLS = ['PROVINCIA', 'IPRESS', 'NIVEL_EESS', 'COD_SERVICIO', 'SEXO', 'GRUPO_EDAD']

RENAME_MAP = {
    'AÑO':          'anio',
    'MES':          'mes',
    'PROVINCIA':    'provincia',
    'IPRESS':       'ipress',
    'NIVEL_EESS':   'nivel_eess',
    'COD_SERVICIO': 'cod_servicio',
    'SEXO':         'sexo',
    'GRUPO_EDAD':   'grupo_edad',
}

# ── 1. Credenciales ──────────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
URL = os.getenv('SUPABASE_URL')
KEY = os.getenv('SUPABASE_KEY')

if not URL or not KEY:
    raise SystemExit("ERROR: SUPABASE_URL o SUPABASE_KEY no encontrados en .env")

def hdrs(schema, returning='minimal'):
    return {
        'apikey':          KEY,
        'Authorization':   f'Bearer {KEY}',
        'Content-Type':    'application/json',
        'Prefer':          f'return={returning}',
        'Content-Profile': schema,
        'Accept-Profile':  schema,
    }

print(f"[1/5] Credenciales cargadas — {URL}")

# ── 2. Leer CSV por chunks, filtrar Cajamarca y agrupar ─────────────────────
base      = os.path.join(os.path.dirname(__file__), '..', 'assets')
csv_files = sorted(glob.glob(os.path.join(base, '*.csv')))

if not csv_files:
    raise SystemExit(f"ERROR: no se encontraron CSV en {base}")

print(f"[2/5] Procesando {len(csv_files)} CSV en chunks de {CHUNK_SIZE:,} filas...\n")

parciales   = []
t_total_ini = time.time()

for csv_idx, path in enumerate(csv_files, 1):
    nombre    = os.path.basename(path)
    acum      = []
    filas_raw = 0
    filas_caj = 0
    n_chunks  = 0
    t_csv     = time.time()

    print(f"  [{csv_idx}/{len(csv_files)}] {nombre}")

    try:
        reader = pd.read_csv(
            path, encoding='utf-8', sep=',',
            dtype=str, chunksize=CHUNK_SIZE,
        )
    except Exception:
        reader = pd.read_csv(
            path, encoding='latin-1', sep=',',
            dtype=str, chunksize=CHUNK_SIZE,
        )

    for chunk in reader:
        n_chunks  += 1
        filas_raw += len(chunk)

        # Normalizar nombre de columna AÑO (por si el CSV trae codificación rota)
        chunk.columns = [
            c.strip().replace('AÃ\x91O', 'AÑO').replace('AÃO', 'AÑO')
            for c in chunk.columns
        ]

        if 'REGION' not in chunk.columns:
            print(f"        [WARN] chunk {n_chunks}: columna REGION no encontrada, saltando")
            continue

        # Filtrar Cajamarca
        mask  = chunk['REGION'].str.strip().str.upper() == 'CAJAMARCA'
        chunk = chunk[mask].copy()
        if chunk.empty:
            continue
        filas_caj += len(chunk)

        # Limpiar texto
        for col in TEXT_COLS:
            if col in chunk.columns:
                chunk[col] = chunk[col].str.strip().str.upper()

        # Tipos numéricos
        chunk['ATENCIONES'] = (
            pd.to_numeric(chunk.get('ATENCIONES'), errors='coerce')
            .fillna(0).astype(int)
        )
        chunk['AÑO'] = pd.to_numeric(chunk.get('AÑO'), errors='coerce').astype('Int16')
        chunk['MES'] = pd.to_numeric(chunk.get('MES'),  errors='coerce').astype('Int16')

        chunk = chunk[chunk['ATENCIONES'] >= 0].dropna(subset=['AÑO', 'MES'])

        # GROUP BY parcial dentro del chunk
        cols_presentes = [c for c in GROUP_COLS if c in chunk.columns]
        grp = (
            chunk.groupby(cols_presentes, as_index=False, dropna=False)
                 .agg(ATENCIONES=('ATENCIONES', 'sum'))
        )
        acum.append(grp)

        if n_chunks % 10 == 0:
            print(f"        chunk {n_chunks:>3} | {filas_raw:>10,} leídas | "
                  f"{filas_caj:>8,} Cajamarca")

    if not acum:
        print(f"        [WARN] ningún registro Cajamarca en este CSV\n")
        continue

    # Reagrupar todo el CSV y acumular
    parcial = (
        pd.concat(acum, ignore_index=True)
          .groupby(GROUP_COLS, as_index=False, dropna=False)
          .agg(ATENCIONES=('ATENCIONES', 'sum'))
    )
    parciales.append(parcial)

    print(f"        Filas totales : {filas_raw:>10,}")
    print(f"        Cajamarca     : {filas_caj:>10,}")
    print(f"        Grupos CSV    : {len(parcial):>10,}")
    print(f"        Tiempo        : {time.time()-t_csv:.1f}s\n")

# ── 3. Combinar los 4 parciales y reagrupar ──────────────────────────────────
print("[3/5] Combinando parciales y reagrupando en Gold...")
gold = (
    pd.concat(parciales, ignore_index=True)
      .groupby(GROUP_COLS, as_index=False, dropna=False)
      .agg(total_atenciones=('ATENCIONES', 'sum'))
)

gold.rename(columns=RENAME_MAP, inplace=True)
gold['anio']             = gold['anio'].astype(int)
gold['mes']              = gold['mes'].astype(int)
gold['total_atenciones'] = gold['total_atenciones'].astype(int)

total_filas = len(gold)
print(f"      Filas Gold finales : {total_filas:,}")
print(f"      Tiempo acumulado   : {time.time()-t_total_ini:.1f}s\n")

# ── 4. Limpiar tabla Gold y registrar inicio en meta_pipeline ────────────────
print("[4/5] Limpiando tabla Gold y registrando en meta_pipeline...")

r = requests.delete(
    f"{URL}/rest/v1/demanda_mensual_ipress",
    headers=hdrs('gold'),
    params={'id': 'gte.0'},
)
print(f"      DELETE gold.demanda_mensual_ipress : HTTP {r.status_code}")
if r.status_code not in (200, 204):
    print(f"      [WARN] {r.text[:200]}")

meta_payload = {
    'tabla':            'gold.demanda_mensual_ipress',
    'tipo_carga':       'bronze_gold_directo',
    'fecha_inicio':     datetime.now().isoformat(),
    'filas_procesadas': total_filas,
    'estado':           'en_proceso',
}
r = requests.post(
    f"{URL}/rest/v1/meta_pipeline",
    headers=hdrs('bronze', returning='representation'),
    data=json.dumps(meta_payload),
)
meta_id = r.json()[0]['id'] if r.status_code in (200, 201) else None
print(f"      meta_pipeline id : {meta_id}\n")

# ── 5. Cargar Gold en batches ────────────────────────────────────────────────
print(f"[5/5] Cargando {total_filas:,} filas en batches de {BATCH_SIZE}...")

records   = gold.to_dict(orient='records')
n_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
errores   = 0
t_carga   = time.time()

for i in range(n_batches):
    batch = records[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    r = requests.post(
        f"{URL}/rest/v1/demanda_mensual_ipress",
        headers=hdrs('gold'),
        data=json.dumps(batch),
    )
    if r.status_code not in (200, 201):
        errores += 1
        print(f"      [ERROR] batch {i+1}: HTTP {r.status_code} — {r.text[:120]}")

    if (i + 1) % 50 == 0 or (i + 1) == n_batches:
        pct      = (i + 1) / n_batches * 100
        elapsed  = time.time() - t_carga
        filas_ok = min((i + 1) * BATCH_SIZE, total_filas)
        vel      = filas_ok / elapsed if elapsed > 0 else 0
        print(f"      Batch {i+1:>5}/{n_batches} ({pct:5.1f}%) | "
              f"{filas_ok:>8,} filas | {elapsed:5.1f}s | {vel:,.0f} filas/s")

# ── 6. Actualizar meta_pipeline ──────────────────────────────────────────────
estado = 'completado' if errores == 0 else f'errores: {errores} batches'
if meta_id:
    requests.patch(
        f"{URL}/rest/v1/meta_pipeline?id=eq.{meta_id}",
        headers={**hdrs('bronze'), 'Prefer': 'return=minimal'},
        data=json.dumps({
            'fecha_fin':     datetime.now().isoformat(),
            'estado':        estado,
            'observaciones': (
                f"{total_filas:,} filas gold | "
                f"{n_batches} batches | "
                f"errores: {errores} | "
                f"{time.time()-t_total_ini:.0f}s total"
            ),
        }),
    )

t_fin = time.time() - t_total_ini
sep = "=" * 55
print(f"\n{sep}")
print(f"  GOLD CARGADO {'EXITOSAMENTE' if errores == 0 else 'CON ERRORES'}")
print(f"  Filas cargadas : {total_filas:,}")
print(f"  Errores        : {errores} batches")
print(f"  Tiempo total   : {t_fin:.1f}s")
print(f"  Estado         : {estado}")
print(sep)
