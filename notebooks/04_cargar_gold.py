import os, json, time
import requests
from datetime import datetime
from dotenv import load_dotenv

# â”€â”€ 1. Credenciales â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
URL = os.getenv('SUPABASE_URL')
KEY = os.getenv('SUPABASE_KEY')

headers_public = {
    'apikey':        KEY,
    'Authorization': f'Bearer {KEY}',
    'Content-Type':  'application/json',
    'Prefer':        'return=representation',
}
headers_bronze = {**headers_public, 'Content-Profile': 'bronze', 'Accept-Profile': 'bronze'}

print("[1/4] Credenciales cargadas")

# â”€â”€ 2. Registrar inicio en meta_pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("[2/4] Registrando inicio en bronze.meta_pipeline...")

meta_inicio = {
    'tabla':        'gold.demanda_mensual_ipress',
    'tipo_carga':   'agregacion_silver_gold',
    'fecha_inicio': datetime.now().isoformat(),
    'estado':       'en_proceso',
}
r = requests.post(
    f"{URL}/rest/v1/meta_pipeline",
    headers=headers_bronze,
    data=json.dumps(meta_inicio)
)
if r.status_code not in (200, 201):
    print(f"  [AVISO] meta_pipeline inicio: {r.status_code} {r.text[:200]}")
    meta_id = None
else:
    meta_id = r.json()[0]['id']
    print(f"  Registro meta_pipeline id={meta_id}")

# â”€â”€ 3. Llamar RPC poblar_gold_demanda â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("[3/4] Ejecutando RPC poblar_gold_demanda() en PostgreSQL...")
print("  (Silver â†’ GROUP BY â†’ Gold, todo dentro de Supabase)")

t0 = time.time()
r = requests.post(
    f"{URL}/rest/v1/rpc/poblar_gold_demanda",
    headers={
        'apikey':        KEY,
        'Authorization': f'Bearer {KEY}',
        'Content-Type':  'application/json',
    },
    data=json.dumps({})
)
elapsed = time.time() - t0

if r.status_code in (200, 201):
    resultado = r.json()
    print(f"  {resultado}")
    estado = 'completado'
    # Extraer nÃºmero de filas del mensaje
    try:
        n_filas = int(''.join(filter(str.isdigit, resultado.split('filas')[0].split(':')[-1].strip())))
    except Exception:
        n_filas = None
else:
    resultado = r.text
    estado = f'error: {r.status_code}'
    n_filas = None
    print(f"  [ERROR] {r.status_code}: {r.text[:300]}")

print(f"  Tiempo RPC: {elapsed:.1f}s")

# â”€â”€ 4. Actualizar meta_pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("[4/4] Actualizando meta_pipeline...")

if meta_id is not None:
    meta_fin = {
        'fecha_fin':        datetime.now().isoformat(),
        'estado':           estado,
        'filas_procesadas': n_filas,
        'observaciones':    str(resultado),
    }
    r2 = requests.patch(
        f"{URL}/rest/v1/meta_pipeline?id=eq.{meta_id}",
        headers={**headers_bronze, 'Prefer': 'return=minimal'},
        data=json.dumps(meta_fin)
    )
    if r2.status_code not in (200, 201, 204):
        print(f"  [AVISO] meta_pipeline update: {r2.status_code}")

print(f"\n{'='*55}")
print(f"  GOLD COMPLETADO")
print(f"  Filas en gold  : {n_filas:,}" if n_filas else "  Filas en gold  : ver Supabase")
print(f"  Tiempo total   : {elapsed:.1f}s")
print(f"  Estado         : {estado}")
print(f"{'='*55}")

