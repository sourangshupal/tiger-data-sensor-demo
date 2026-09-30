-- Schema for the Agentic-Postgres sensor demo (Tiger Cloud)
-- Sensors metadata + sensor_data hypertable + hourly continuous aggregate

CREATE TABLE IF NOT EXISTS sensors (
    sensor_id TEXT PRIMARY KEY,
    model     TEXT,
    location  TEXT
);

CREATE TABLE IF NOT EXISTS sensor_data (
    "timestamp" TIMESTAMPTZ      NOT NULL,
    sensor_id   TEXT             NOT NULL,
    temperature DOUBLE PRECISION,
    humidity    DOUBLE PRECISION
);

SELECT create_hypertable('sensor_data', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS sensor_data_sensor_time_idx
    ON sensor_data (sensor_id, "timestamp" DESC);

-- Hourly roll-up per sensor (continuous aggregate, real-time mode by default:
-- recent data not yet materialized is computed from the raw hypertable on the fly)
CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', "timestamp") AS hour,
    sensor_id,
    avg(temperature) AS avg_temp,
    avg(humidity)    AS avg_humidity,
    min(temperature) AS min_temp,
    max(temperature) AS max_temp,
    count(*)         AS readings
FROM sensor_data
GROUP BY hour, sensor_id;
