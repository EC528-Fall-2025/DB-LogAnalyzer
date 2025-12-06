create table if not exists events (
  event_id          varchar primary key,         
  ts                timestamp not null,         
  severity          smallint,                  
  event             varchar not null,            
  process           varchar,                     
  role              varchar,                     
  pid               integer,                     
  machine_id        varchar,                   
  address           varchar,                   
  trace_file        varchar,                   
  src_line          integer,                     
  raw_json          json,                        
  fields_json       json                        
);

create table if not exists event_metrics (
  event_id     varchar not null,
  event        varchar,
  metric_name  varchar not null,
  metric_value double,
  unit         varchar,            
  is_counter   boolean default false,
  primary key (event_id, metric_name)
);

create table if not exists metric_baselines (
  metric_name varchar not null,
  role        varchar,
  mean        double,
  stddev      double,
  p95         double,
  min         double,
  max         double,
  count       bigint,
  updated_at  timestamp,
  primary key (metric_name, role)
);

create table if not exists events_wide (
  event_id            varchar(255) primary key references events(event_id),
  grv_latency_ms      double,
  txn_volume          double,
  queue_bytes         double,
  durability_lag_s    double,
  data_move_in_flight double,
  disk_queue_bytes    double,
  kv_ops              double
);

create table if not exists processes (
  process_key   varchar primary key,    
  first_seen_ts timestamp,
  last_seen_ts  timestamp,
  address       varchar,
  pid           integer,
  class         varchar,                
  version       varchar,               
  command_line  varchar
);

create table if not exists process_roles (
  process_key   varchar references processes(process_key),
  role          varchar,
  start_ts      timestamp,
  end_ts        timestamp,             
  primary key (process_key, role, start_ts)
);
