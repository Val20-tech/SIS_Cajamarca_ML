import os
import psycopg2
from dotenv import load_dotenv

# --- 1. Cargar credenciales ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

host     = os.getenv('SUPABASE_DB_HOST')
password = os.getenv('SUPABASE_DB_PASSWORD')
user     = os.getenv('SUPABASE_DB_USER')
dbname   = os.getenv('SUPABASE_DB_NAME')
port     = os.getenv('SUPABASE_DB_PORT')

print(f"[1/4] Credenciales cargadas — host: {host}")

# --- 2. Leer SQL ---
sql_path = os.path.join(os.path.dirname(__file__), '..', 'sql', 'bronze', '01_crear_esquemas.sql')
with open(sql_path, 'r', encoding='utf-8') as f:
    sql = f.read()
print(f"[2/4] SQL leído desde {os.path.normpath(sql_path)}")

# --- 3. Conectar y ejecutar ---
conn = None
try:
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        sslmode='require',
        connect_timeout=10
    )
    print("[3/4] Conexión a Supabase exitosa")

    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print("[4/4] SQL ejecutado correctamente — esquemas bronze, silver, gold y tabla meta_pipeline creados")

except Exception as e:
    print(f"[ERROR] {e}")
    if conn:
        conn.rollback()

finally:
    if conn:
        conn.close()
        print("[OK] Conexión cerrada")
