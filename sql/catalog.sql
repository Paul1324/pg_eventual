CREATE TYPE eventual.derived;
CREATE FUNCTION eventual.derived_in(cstring) RETURNS eventual.derived
    LANGUAGE internal IMMUTABLE STRICT PARALLEL SAFE AS 'textin';
CREATE FUNCTION eventual.derived_out(eventual.derived) RETURNS cstring
    LANGUAGE internal IMMUTABLE STRICT PARALLEL SAFE AS 'textout';
CREATE FUNCTION eventual.derived_typmod_in(cstring[]) RETURNS integer
    LANGUAGE c IMMUTABLE STRICT AS 'MODULE_PATHNAME', 'eventual_derived_typmod_in';
CREATE TYPE eventual.derived (
    INPUT = eventual.derived_in,
    OUTPUT = eventual.derived_out,
    TYPMOD_IN = eventual.derived_typmod_in,
    LIKE = text
);
COMMENT ON TYPE eventual.derived IS
    'Declaration type for derived columns: eventual.derived(processor, input_column [, ...] [, ''{"param": value}''])';

CREATE TABLE eventual.processors (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE CHECK (name ~ '^[a-z_][a-z0-9_]*$'),
    kind text NOT NULL CHECK (kind IN ('sql', 'triton', 'http')),
    input_types regtype[] NOT NULL CHECK (cardinality(input_types) > 0),
    result_type regtype NOT NULL,
    function regprocedure,
    config jsonb NOT NULL DEFAULT '{}',
    defaults jsonb NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(defaults) = 'object'),
    strict boolean NOT NULL DEFAULT true,
    max_attempts integer NOT NULL DEFAULT 3 CHECK (max_attempts >= 1),
    description text,
    version integer NOT NULL DEFAULT 1,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK ((kind = 'sql') = (function IS NOT NULL))
);
COMMENT ON TABLE eventual.processors IS 'Named, reusable processors that compute derived column values.';

CREATE TABLE eventual.derived_columns (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    relid regclass NOT NULL,
    attnum int2 NOT NULL CHECK (attnum > 0),
    processor_id bigint NOT NULL REFERENCES eventual.processors (id),
    input_attnums int2[] NOT NULL CHECK (cardinality(input_attnums) > 0),
    target_name name NOT NULL,
    input_names name[] NOT NULL,
    params jsonb NOT NULL DEFAULT '{}',
    effective_params jsonb NOT NULL DEFAULT '{}',
    depth integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (relid, attnum)
);
COMMENT ON TABLE eventual.derived_columns IS 'One row per derived column: target, inputs, processor and parameters.';

CREATE TABLE eventual.cells (
    column_id bigint NOT NULL REFERENCES eventual.derived_columns (id) ON DELETE CASCADE,
    row_key text[] NOT NULL,
    state text NOT NULL CHECK (state IN ('waiting', 'pending', 'running', 'done', 'failed')),
    generation bigint NOT NULL DEFAULT 1,
    attempts integer NOT NULL DEFAULT 0,
    enqueued_at timestamptz NOT NULL DEFAULT now(),
    next_attempt_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    finished_at timestamptz,
    duration_ms double precision,
    worker_pid integer,
    last_error text,
    result_meta jsonb,
    PRIMARY KEY (column_id, row_key)
);
CREATE INDEX cells_pending_idx ON eventual.cells (next_attempt_at, enqueued_at) WHERE state = 'pending';
CREATE INDEX cells_running_idx ON eventual.cells (worker_pid) WHERE state = 'running';
COMMENT ON TABLE eventual.cells IS 'Computation state of each derived value (waiting, pending, running, done, failed).';

CREATE TABLE eventual.secrets (
    name text PRIMARY KEY CHECK (name ~ '^[A-Za-z_][A-Za-z0-9_]*$'),
    value text NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE eventual.control (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    paused boolean NOT NULL DEFAULT false,
    updated_at timestamptz NOT NULL DEFAULT now()
);
INSERT INTO eventual.control DEFAULT VALUES;

CREATE TABLE eventual.workers (
    worker_index integer PRIMARY KEY,
    pid integer,
    state text NOT NULL,
    started_at timestamptz,
    heartbeat_at timestamptz,
    processed bigint NOT NULL DEFAULT 0,
    failed bigint NOT NULL DEFAULT 0,
    last_error text
);
COMMENT ON TABLE eventual.workers IS 'Heartbeat and counters reported by the background workers.';
