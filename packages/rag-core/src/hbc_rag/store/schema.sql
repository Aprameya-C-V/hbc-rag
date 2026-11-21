create schema if not exists hbc_rag;
create extension if not exists vector with schema public;
set search_path to hbc_rag, public;

create table if not exists documents (
  id        bigserial primary key,
  dataset   text not null,
  doc_key   text not null unique,
  title     text not null,
  body      text not null,
  tsv       tsvector generated always as (to_tsvector('english', title || ' ' || body)) stored,
  embedding vector(384) not null
);

create index if not exists documents_embedding_hnsw
  on documents using hnsw (embedding vector_cosine_ops);
create index if not exists documents_tsv_gin
  on documents using gin (tsv);
create index if not exists documents_dataset_idx
  on documents (dataset);

create table if not exists runs (
  id          text primary key,
  model       text not null,
  provider    text not null,
  policy      text not null,
  dataset     text not null,
  created_at  timestamptz not null default now(),
  metrics     jsonb not null default '{}'::jsonb,
  tags        jsonb not null default '{}'::jsonb
);

create table if not exists hops (
  id                bigserial primary key,
  run_id            text not null references runs(id) on delete cascade,
  question_id       text not null default '',
  hop_index         int  not null,
  query             text not null,
  retrieved         jsonb not null default '[]'::jsonb,
  prompt_tokens     int  not null default 0,
  cached_tokens     int  not null default 0,
  completion_tokens int  not null default 0,
  latency_ms        double precision not null default 0,
  decision          text not null default 'answer'
);

create index if not exists hops_run_idx on hops (run_id);
