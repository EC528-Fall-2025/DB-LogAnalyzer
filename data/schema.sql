CREATE TABLE IF NOT EXISTS events (
  event_id    INTEGER,
  ts          TIMESTAMP,
  severity    INTEGER,
  event       VARCHAR,
  process     VARCHAR,
  role        VARCHAR,
  pid         INTEGER,
  machine_id  VARCHAR,
  address     VARCHAR,
  trace_file  VARCHAR,
  src_line    INTEGER,
  raw_json    JSON,
  fields_json JSON
);
