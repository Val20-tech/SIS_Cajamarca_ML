CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

CREATE TABLE IF NOT EXISTS bronze.meta_pipeline (
    id SERIAL PRIMARY KEY,
    tabla TEXT NOT NULL,
    tipo_carga TEXT NOT NULL,
    fecha_inicio TIMESTAMP DEFAULT NOW(),
    fecha_fin TIMESTAMP,
    filas_procesadas INTEGER,
    estado TEXT,
    observaciones TEXT
);
