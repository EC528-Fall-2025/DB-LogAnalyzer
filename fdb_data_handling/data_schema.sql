create table if not exists events (
  event_id     bigint primary key,
  ts           timestamp not null,
  severity     smallint,
  event        varchar not null,
  process      varchar,
  role         varchar,
  pid          integer,
  machine_id   varchar,
  address      varchar,
  trace_file   varchar,
  src_line     integer,
  raw_json     json,
  fields_json  json
);

create table if not exists event_metrics (
  event_id     bigint not null,
  metric_name  varchar not null,
  metric_value double,
  unit         varchar,
  is_counter   boolean default false,
  primary key (event_id, metric_name)
);