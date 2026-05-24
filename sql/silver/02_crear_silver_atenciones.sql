-- Tabla principal Silver: atenciones limpias de Cajamarca
CREATE TABLE IF NOT EXISTS silver.atenciones_cajamarca (
    id          BIGSERIAL PRIMARY KEY,
    anio        SMALLINT    NOT NULL,
    mes         SMALLINT    NOT NULL,
    region      TEXT        NOT NULL,
    provincia   TEXT,
    distrito    TEXT,
    ipress      TEXT,
    nivel_eess  TEXT,
    plan_seguro TEXT,
    cod_servicio    TEXT,
    desc_servicio   TEXT,
    sexo            TEXT,
    grupo_edad      TEXT,
    atenciones      INTEGER  NOT NULL DEFAULT 0,
    cargado_en      TIMESTAMP DEFAULT NOW()
);

-- Sin RLS: carga via API REST con anon key
ALTER TABLE silver.atenciones_cajamarca DISABLE ROW LEVEL SECURITY;

-- Permisos para el rol anon (necesario para REST API)
GRANT USAGE  ON SCHEMA silver TO anon;
GRANT ALL    ON TABLE  silver.atenciones_cajamarca TO anon;
GRANT USAGE, SELECT ON SEQUENCE silver.atenciones_cajamarca_id_seq TO anon;

-- bronze.meta_pipeline: mismos permisos
ALTER TABLE bronze.meta_pipeline DISABLE ROW LEVEL SECURITY;
GRANT USAGE  ON SCHEMA bronze TO anon;
GRANT ALL    ON TABLE  bronze.meta_pipeline TO anon;
GRANT USAGE, SELECT ON SEQUENCE bronze.meta_pipeline_id_seq TO anon;
