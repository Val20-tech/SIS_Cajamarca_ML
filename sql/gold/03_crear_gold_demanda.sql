-- ── Tabla Gold: demanda agregada por IPRESS ──────────────────────────────────
CREATE TABLE IF NOT EXISTS gold.demanda_mensual_ipress (
    id               BIGSERIAL PRIMARY KEY,
    anio             SMALLINT NOT NULL,
    mes              SMALLINT NOT NULL,
    region           TEXT     NOT NULL,
    provincia        TEXT,
    distrito         TEXT,
    ipress           TEXT,
    nivel_eess       TEXT,
    plan_seguro      TEXT,
    cod_servicio     TEXT,
    desc_servicio    TEXT,
    sexo             TEXT,
    grupo_edad       TEXT,
    total_atenciones INTEGER  NOT NULL DEFAULT 0,
    cargado_en       TIMESTAMP DEFAULT NOW()
);

-- Índices para consultas del modelo
CREATE INDEX IF NOT EXISTS idx_gold_anio_mes
    ON gold.demanda_mensual_ipress (anio, mes);
CREATE INDEX IF NOT EXISTS idx_gold_ipress
    ON gold.demanda_mensual_ipress (ipress);
CREATE INDEX IF NOT EXISTS idx_gold_servicio
    ON gold.demanda_mensual_ipress (cod_servicio);

-- Sin RLS + permisos anon
ALTER TABLE gold.demanda_mensual_ipress DISABLE ROW LEVEL SECURITY;
GRANT USAGE  ON SCHEMA gold TO anon;
GRANT ALL    ON TABLE  gold.demanda_mensual_ipress TO anon;
GRANT USAGE, SELECT ON SEQUENCE gold.demanda_mensual_ipress_id_seq TO anon;

-- ── Función RPC: agrega Silver → Gold en PostgreSQL ───────────────────────────
CREATE OR REPLACE FUNCTION public.poblar_gold_demanda()
RETURNS TEXT AS $$
DECLARE
    n_rows INTEGER;
BEGIN
    SET LOCAL statement_timeout = 0;
    TRUNCATE TABLE gold.demanda_mensual_ipress;

    INSERT INTO gold.demanda_mensual_ipress (
        anio, mes, region, provincia, distrito,
        ipress, nivel_eess, plan_seguro,
        cod_servicio, desc_servicio,
        sexo, grupo_edad, total_atenciones
    )
    SELECT
        anio, mes, region, provincia, distrito,
        ipress, nivel_eess, plan_seguro,
        cod_servicio, desc_servicio,
        sexo, grupo_edad,
        SUM(atenciones) AS total_atenciones
    FROM silver.atenciones_cajamarca
    GROUP BY
        anio, mes, region, provincia, distrito,
        ipress, nivel_eess, plan_seguro,
        cod_servicio, desc_servicio,
        sexo, grupo_edad;

    GET DIAGNOSTICS n_rows = ROW_COUNT;
    RETURN 'Completado: ' || n_rows || ' filas insertadas en gold.demanda_mensual_ipress';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Permitir que el rol anon llame la función
GRANT EXECUTE ON FUNCTION public.poblar_gold_demanda() TO anon;
