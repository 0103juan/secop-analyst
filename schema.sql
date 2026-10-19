-- Applied by `python db.py` and safe to run again. The embedding size is that of the model in db.py.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS contracts (
    id            text PRIMARY KEY,           -- id_contrato in SECOP II
    entity        text NOT NULL,
    entity_nit    bigint,
    department    text,
    city          text,
    contract_type text,
    modality      text,
    status        text,
    object        text NOT NULL,              -- what the contract is for, free text
    supplier      text,
    signed_on     date,
    year          int GENERATED ALWAYS AS (extract(year FROM signed_on)::int) STORED,
    value_cop     numeric(20, 0),
    updated_at    timestamptz,                -- ultima_actualizacion upstream: the cursor of the incremental sync
    url           text,
    object_tsv    tsvector GENERATED ALWAYS AS (to_tsvector('spanish', object)) STORED,
    embedding     vector(384)
);
CREATE INDEX IF NOT EXISTS contracts_embedding ON contracts USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS contracts_object_tsv ON contracts USING gin (object_tsv);
CREATE INDEX IF NOT EXISTS contracts_entity_signed ON contracts (entity_nit, signed_on);
CREATE INDEX IF NOT EXISTS contracts_updated ON contracts (updated_at);

-- Long-term memory: what the agent keeps about a user after it proved useful more than once.
CREATE TABLE IF NOT EXISTS memories (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      text NOT NULL,
    text         text NOT NULL,
    embedding    vector(384) NOT NULL,
    strength     int NOT NULL DEFAULT 1,      -- times it was recalled or seen again
    created_at   timestamptz NOT NULL DEFAULT now(),
    last_used_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS memories_user ON memories (user_id, last_used_at);

-- Who may call the API: one row per key, kept as a hash (keys.py).
CREATE TABLE IF NOT EXISTS api_keys (
    key_hash   bytea PRIMARY KEY,
    user_id    text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    revoked_at timestamptz
);

-- The only role that runs SQL written by the model: it can read one table and nothing else.
DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'analyst_ro') THEN
        CREATE ROLE analyst_ro NOLOGIN;
    END IF;
END $$;
GRANT SELECT ON contracts TO analyst_ro;

-- Where each slice of the upstream dataset was left: the cursor of ingest.py.
CREATE TABLE IF NOT EXISTS sync_state (
    slice  text PRIMARY KEY,                 -- the filter the rows were asked with
    newest timestamptz NOT NULL
);
