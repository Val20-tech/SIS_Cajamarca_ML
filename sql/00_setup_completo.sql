-- ═══════════════════════════════════════════════════════════════════════════
-- SIS Cajamarca ML — Setup completo en nuevo Supabase
-- Ejecutar completo en SQL Editor de Supabase (una sola vez)
-- Arquitectura Medallion: Bronze y Silver NO existen en Supabase
-- Solo Gold se persiste en la base de datos
-- ═══════════════════════════════════════════════════════════════════════════

-- ── 1. Esquemas ─────────────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

-- ── 2. bronze.meta_pipeline (log de cargas) ─────────────────────────────────
CREATE TABLE IF NOT EXISTS bronze.meta_pipeline (
    id                SERIAL      PRIMARY KEY,
    tabla             TEXT        NOT NULL,
    tipo_carga        TEXT        NOT NULL,
    fecha_inicio      TIMESTAMP   DEFAULT NOW(),
    fecha_fin         TIMESTAMP,
    filas_procesadas  INTEGER,
    estado            TEXT,
    observaciones     TEXT
);

ALTER TABLE bronze.meta_pipeline DISABLE ROW LEVEL SECURITY;
GRANT USAGE  ON SCHEMA bronze TO anon;
GRANT ALL    ON TABLE  bronze.meta_pipeline TO anon;
GRANT USAGE, SELECT ON SEQUENCE bronze.meta_pipeline_id_seq TO anon;

-- ── 3. gold.demanda_mensual_ipress (tabla objetivo) ─────────────────────────
-- GROUP BY: anio, mes, provincia, ipress, nivel_eess, cod_servicio, sexo, grupo_edad
CREATE TABLE IF NOT EXISTS gold.demanda_mensual_ipress (
    id                BIGSERIAL   PRIMARY KEY,
    anio              SMALLINT    NOT NULL,
    mes               SMALLINT    NOT NULL,
    provincia         TEXT,
    ipress            TEXT,
    nivel_eess        TEXT,
    cod_servicio      TEXT,
    sexo              TEXT,
    grupo_edad        TEXT,
    total_atenciones  INTEGER     NOT NULL DEFAULT 0,
    cargado_en        TIMESTAMP   DEFAULT NOW()
);

-- Índices para consultas del modelo predictivo
CREATE INDEX IF NOT EXISTS idx_gold_anio_mes
    ON gold.demanda_mensual_ipress (anio, mes);
CREATE INDEX IF NOT EXISTS idx_gold_ipress
    ON gold.demanda_mensual_ipress (ipress);
CREATE INDEX IF NOT EXISTS idx_gold_provincia
    ON gold.demanda_mensual_ipress (provincia);
CREATE INDEX IF NOT EXISTS idx_gold_servicio
    ON gold.demanda_mensual_ipress (cod_servicio);

ALTER TABLE gold.demanda_mensual_ipress DISABLE ROW LEVEL SECURITY;
GRANT USAGE  ON SCHEMA gold TO anon;
GRANT ALL    ON TABLE  gold.demanda_mensual_ipress TO anon;
GRANT USAGE, SELECT ON SEQUENCE gold.demanda_mensual_ipress_id_seq TO anon;

-- ── Verificación final ───────────────────────────────────────────────────────
SELECT
    schemaname,
    tablename
FROM pg_tables
WHERE schemaname IN ('bronze', 'silver', 'gold')
ORDER BY schemaname, tablename;
