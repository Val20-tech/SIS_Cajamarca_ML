-- ── gold.ml_predicciones — predicciones del test set (20%) ──────────────────
-- Ejecutar en Supabase SQL Editor una sola vez
CREATE TABLE IF NOT EXISTS gold.ml_predicciones (
    id              BIGSERIAL   PRIMARY KEY,
    anio            SMALLINT,
    mes             SMALLINT,
    provincia       TEXT,
    ipress          TEXT,
    nivel_eess      TEXT,
    cod_servicio    TEXT,
    sexo            TEXT,
    grupo_edad      TEXT,
    atenciones_real INTEGER,
    atenciones_pred NUMERIC(10,2),
    error_abs       NUMERIC(10,2),
    cargado_en      TIMESTAMP   DEFAULT NOW()
);

ALTER TABLE gold.ml_predicciones DISABLE ROW LEVEL SECURITY;
GRANT USAGE  ON SCHEMA gold TO anon;
GRANT ALL    ON TABLE  gold.ml_predicciones TO anon;
GRANT USAGE, SELECT ON SEQUENCE gold.ml_predicciones_id_seq TO anon;
