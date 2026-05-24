import os
import glob
import pandas as pd
from dotenv import load_dotenv

# --- 1. Credenciales ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
print("[1/4] Credenciales cargadas")

# --- 2. Leer y filtrar los 4 CSV ---
base = os.path.join(os.path.dirname(__file__), '..', 'assets')
csv_files = sorted(glob.glob(os.path.join(base, '*.csv')))

if not csv_files:
    raise FileNotFoundError("No se encontraron CSV en assets/")

print(f"[2/4] Archivos encontrados: {len(csv_files)}")
for f in csv_files:
    print(f"      {os.path.basename(f)}")

frames = []
for path in csv_files:
    df = pd.read_csv(path, encoding='utf-8', sep=',', dtype=str)
    antes = len(df)
    df = df[df['REGION'].str.strip().str.upper() == 'CAJAMARCA']
    print(f"      {os.path.basename(path)}: {antes:,} filas totales -> {len(df):,} Cajamarca")
    frames.append(df)

# --- 3. Concatenar ---
data = pd.concat(frames, ignore_index=True)
print(f"\n[3/4] Concatenación completa")

# --- 4. Estadísticas ---
print(f"\n{'='*55}")
print(f"  TOTAL FILAS : {len(data):,}")
print(f"  COLUMNAS    : {len(data.columns)}")
print(f"{'='*55}")
print(f"\nColumnas disponibles:")
for col in data.columns:
    print(f"  · {col}")

print(f"\nMuestra (5 filas):")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
print(data.head(5).to_string(index=False))
print(f"\n[4/4] Listo — sin carga a Supabase")
