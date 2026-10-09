CREATE FUNCTION eventual._row_trigger() RETURNS trigger
    LANGUAGE c AS 'MODULE_PATHNAME', 'eventual_row_trigger';

CREATE FUNCTION eventual._flush_trigger() RETURNS trigger
    LANGUAGE c SECURITY DEFINER SET search_path = pg_catalog, pg_temp
    AS 'MODULE_PATHNAME', 'eventual_flush_trigger';

CREATE FUNCTION eventual._truncate_trigger() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $$
BEGIN
    DELETE FROM eventual.cells
    WHERE column_id IN (SELECT id FROM eventual.derived_columns WHERE relid = TG_RELID::regclass);
    RETURN NULL;
END
$$;

CREATE FUNCTION eventual.create_processor(
    name text,
    kind text DEFAULT 'sql',
    function regprocedure DEFAULT NULL,
    input_types regtype[] DEFAULT NULL,
    result_type regtype DEFAULT NULL,
    config jsonb DEFAULT '{}',
    defaults jsonb DEFAULT '{}',
    strict boolean DEFAULT true,
    max_attempts integer DEFAULT 3,
    description text DEFAULT NULL
) RETURNS bigint
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $$
DECLARE
    v_proc pg_proc;
    v_inputs regtype[];
    v_result regtype;
    v_id bigint;
BEGIN
    IF name IS NULL OR name !~ '^[a-z_][a-z0-9_]*$' THEN
        RAISE EXCEPTION 'invalid processor name "%"', name
            USING HINT = 'Processor names are lower-case identifiers so they can be written unquoted in eventual.derived(...).';
    END IF;
    IF defaults IS NULL OR jsonb_typeof(defaults) <> 'object' THEN
        RAISE EXCEPTION 'processor defaults must be a JSON object';
    END IF;
    IF EXISTS (SELECT 1 FROM eventual.processors p WHERE p.name = create_processor.name) THEN
        RAISE EXCEPTION 'processor "%" already exists', name USING ERRCODE = 'duplicate_object';
    END IF;

    IF kind = 'sql' THEN
        IF function IS NULL THEN
            RAISE EXCEPTION 'sql processors need a function (for example function => ''my_fn(text)''::regprocedure)';
        END IF;
        SELECT * INTO v_proc FROM pg_proc WHERE oid = function;
        IF v_proc.prokind <> 'f' THEN
            RAISE EXCEPTION '% is not a plain function', function;
        END IF;
        IF v_proc.proretset THEN
            RAISE EXCEPTION 'processor function % must not return a set', function;
        END IF;
        SELECT array_agg(t::regtype ORDER BY o) INTO v_inputs
        FROM unnest(v_proc.proargtypes::oid[]) WITH ORDINALITY AS a(t, o);
        IF defaults <> '{}' THEN
            IF coalesce(cardinality(v_inputs), 0) < 2 OR v_inputs[cardinality(v_inputs)] <> 'jsonb'::regtype THEN
                RAISE EXCEPTION 'processor "%" has parameters, so % must take the effective parameters as a trailing jsonb argument', name, function;
            END IF;
            v_inputs := v_inputs[1:cardinality(v_inputs) - 1];
        END IF;
        IF coalesce(cardinality(v_inputs), 0) = 0 THEN
            RAISE EXCEPTION 'processor function % must take at least one input argument', function;
        END IF;
        v_result := v_proc.prorettype::regtype;
        IF (SELECT typtype FROM pg_type WHERE oid = v_result) = 'p' THEN
            RAISE EXCEPTION 'processor function % must return a concrete type, not %', function, v_result;
        END IF;
        IF input_types IS NOT NULL AND input_types <> v_inputs THEN
            RAISE EXCEPTION 'input_types % do not match the function arguments %', input_types, v_inputs;
        END IF;
        IF result_type IS NOT NULL AND result_type <> v_result THEN
            RAISE EXCEPTION 'result_type % does not match the function result %', result_type, v_result;
        END IF;
        IF config IS NOT NULL AND config <> '{}' THEN
            RAISE EXCEPTION 'sql processors do not take a config';
        END IF;
    ELSIF kind IN ('triton', 'http') THEN
        IF function IS NOT NULL THEN
            RAISE EXCEPTION '% processors do not take a function', kind;
        END IF;
        IF input_types IS NULL OR cardinality(input_types) = 0 OR result_type IS NULL THEN
            RAISE EXCEPTION '% processors need input_types and result_type', kind;
        END IF;
        IF kind = 'triton' THEN
            PERFORM eventual._validate_triton_config(config, cardinality(input_types), defaults);
        ELSE
            PERFORM eventual._validate_http_config(config, cardinality(input_types), defaults);
        END IF;
        v_inputs := input_types;
        v_result := result_type;
    ELSE
        RAISE EXCEPTION 'unknown processor kind "%" (supported: sql, triton, http)', kind;
    END IF;

    INSERT INTO eventual.processors
        (name, kind, input_types, result_type, function, config, defaults, strict, max_attempts, description)
    VALUES
        (name, kind, v_inputs, v_result, function, coalesce(config, '{}'), defaults, strict, max_attempts, description)
    RETURNING id INTO v_id;
    RETURN v_id;
END
$$;

CREATE FUNCTION eventual.alter_processor(
    name text,
    config jsonb DEFAULT NULL,
    defaults jsonb DEFAULT NULL,
    function regprocedure DEFAULT NULL,
    strict boolean DEFAULT NULL,
    max_attempts integer DEFAULT NULL,
    description text DEFAULT NULL,
    refresh boolean DEFAULT false
) RETURNS integer
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $$
DECLARE
    v_old eventual.processors;
    v_config jsonb;
    v_defaults jsonb;
    v_proc pg_proc;
    v_args regtype[];
    v_column record;
    v_table regclass;
    v_was_strict boolean;
BEGIN
    SELECT * INTO v_old FROM eventual.processors p WHERE p.name = alter_processor.name FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'processor "%" does not exist', name USING ERRCODE = 'undefined_object';
    END IF;
    SELECT p.strict AS was_strict INTO v_was_strict FROM eventual.processors p WHERE p.id = v_old.id;
    v_config := coalesce(config, v_old.config);
    v_defaults := coalesce(defaults, v_old.defaults);
    IF jsonb_typeof(v_defaults) <> 'object' THEN
        RAISE EXCEPTION 'processor defaults must be a JSON object';
    END IF;

    IF v_old.kind = 'sql' THEN
        IF config IS NOT NULL AND config <> '{}' THEN
            RAISE EXCEPTION 'sql processors do not take a config';
        END IF;
        IF function IS NOT NULL OR defaults IS NOT NULL THEN
            SELECT * INTO v_proc FROM pg_proc WHERE oid = coalesce(function, v_old.function);
            SELECT array_agg(t::regtype ORDER BY o) INTO v_args
            FROM unnest(v_proc.proargtypes::oid[]) WITH ORDINALITY AS a(t, o);
            IF v_proc.prorettype::regtype <> v_old.result_type THEN
                RAISE EXCEPTION 'the new function returns %, but processor "%" returns %',
                    v_proc.prorettype::regtype, name, v_old.result_type;
            END IF;
            IF v_args IS DISTINCT FROM (CASE WHEN v_defaults <> '{}' THEN v_old.input_types || 'jsonb'::regtype
                                             ELSE v_old.input_types END) THEN
                RAISE EXCEPTION 'function % must take (%)', coalesce(function, v_old.function),
                    array_to_string(CASE WHEN v_defaults <> '{}' THEN v_old.input_types || 'jsonb'::regtype
                                         ELSE v_old.input_types END, ', ');
            END IF;
        END IF;
    ELSE
        IF function IS NOT NULL THEN
            RAISE EXCEPTION '% processors do not take a function', v_old.kind;
        END IF;
        IF v_old.kind = 'triton' THEN
            PERFORM eventual._validate_triton_config(v_config, cardinality(v_old.input_types), v_defaults);
        ELSE
            PERFORM eventual._validate_http_config(v_config, cardinality(v_old.input_types), v_defaults);
        END IF;
    END IF;

    UPDATE eventual.processors p
    SET config = v_config,
        defaults = v_defaults,
        function = coalesce(alter_processor.function, p.function),
        strict = coalesce(alter_processor.strict, p.strict),
        max_attempts = coalesce(alter_processor.max_attempts, p.max_attempts),
        description = coalesce(alter_processor.description, p.description),
        version = p.version + 1,
        updated_at = now()
    WHERE p.id = v_old.id;

    FOR v_column IN SELECT dc.id, dc.params FROM eventual.derived_columns dc WHERE dc.processor_id = v_old.id LOOP
        UPDATE eventual.derived_columns
        SET effective_params = eventual._merge_params(v_defaults, v_column.params)
        WHERE id = v_column.id;
    END LOOP;
    IF alter_processor.strict IS NOT NULL AND alter_processor.strict <> v_was_strict THEN
        FOR v_table IN SELECT DISTINCT dc.relid FROM eventual.derived_columns dc WHERE dc.processor_id = v_old.id LOOP
            PERFORM eventual._finalize_table(v_table);
        END LOOP;
    END IF;
    IF refresh THEN
        PERFORM eventual._enqueue_all(dc.id) FROM eventual.derived_columns dc WHERE dc.processor_id = v_old.id;
    END IF;
    RETURN v_old.version + 1;
END
$$;

CREATE FUNCTION eventual.drop_processor(name text) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $$
DECLARE
    v_id bigint;
    v_users text;
BEGIN
    SELECT id INTO v_id FROM eventual.processors p WHERE p.name = drop_processor.name;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'processor "%" does not exist', name USING ERRCODE = 'undefined_object';
    END IF;
    SELECT string_agg(format('%s.%I', dc.relid, a.attname), ', ') INTO v_users
    FROM eventual.derived_columns dc
    JOIN pg_attribute a ON a.attrelid = dc.relid AND a.attnum = dc.attnum
    WHERE dc.processor_id = v_id;
    IF v_users IS NOT NULL THEN
        RAISE EXCEPTION 'processor "%" is used by derived column(s) %', name, v_users
            USING ERRCODE = 'dependent_objects_still_exist';
    END IF;
    DELETE FROM eventual.processors WHERE id = v_id;
END
$$;

CREATE FUNCTION eventual._input_compatible(have regtype, want regtype, kind text) RETURNS boolean
LANGUAGE sql STABLE AS $$
    SELECT have = want
        OR (SELECT typtype = 'p' FROM pg_type WHERE oid = want)
        OR EXISTS (SELECT 1 FROM pg_cast WHERE castsource = have AND casttarget = want AND castcontext = 'i')
        OR EXISTS (SELECT 1 FROM pg_type WHERE oid = have AND typtype = 'd' AND typbasetype = want)
        OR (kind <> 'sql' AND EXISTS (
                SELECT 1 FROM pg_type h JOIN pg_type he ON he.oid = h.typelem,
                              pg_type w JOIN pg_type we ON we.oid = w.typelem
                WHERE h.oid = have AND w.oid = want AND h.typcategory = 'A' AND w.typcategory = 'A'
                  AND he.typcategory = 'N' AND we.typcategory = 'N'))
$$;

CREATE FUNCTION eventual._bind_column(
    p_relid regclass, p_column name, p_processor text, p_inputs text[], p_params jsonb
) RETURNS bigint
LANGUAGE plpgsql AS $$
DECLARE
    v_proc eventual.processors;
    v_rel record;
    v_target record;
    v_att record;
    v_inputs int2[] := '{}';
    v_id bigint;
BEGIN
    SELECT * INTO v_proc FROM eventual.processors WHERE name = p_processor;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'processor "%" does not exist', p_processor USING ERRCODE = 'undefined_object';
    END IF;
    SELECT relkind, relpersistence INTO v_rel FROM pg_class WHERE oid = p_relid;
    IF v_rel.relkind NOT IN ('r', 'p') THEN
        RAISE EXCEPTION 'derived columns require an ordinary or partitioned table; % is not one', p_relid;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_inherits WHERE inhrelid = p_relid) THEN
        RAISE EXCEPTION 'declare derived columns on the root of a partitioned table, not on % (a partition or child table)', p_relid;
    END IF;
    IF v_rel.relpersistence = 't' THEN
        RAISE EXCEPTION 'derived columns are not supported on temporary tables';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_index WHERE indrelid = p_relid AND indisprimary) THEN
        RAISE EXCEPTION 'table % needs a primary key to have derived columns', p_relid
            USING HINT = 'Computed values are written back to the row identified by its primary key.';
    END IF;

    SELECT attnum, atttypid, attgenerated, attidentity INTO v_target
    FROM pg_attribute WHERE attrelid = p_relid AND attname = p_column AND attnum > 0 AND NOT attisdropped;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'column "%" of % does not exist', p_column, p_relid USING ERRCODE = 'undefined_column';
    END IF;
    IF v_target.atttypid <> v_proc.result_type THEN
        RAISE EXCEPTION 'column "%" has type % but processor "%" returns %',
            p_column, v_target.atttypid::regtype, p_processor, v_proc.result_type USING ERRCODE = 'datatype_mismatch';
    END IF;
    IF v_target.attgenerated <> '' OR v_target.attidentity <> '' THEN
        RAISE EXCEPTION 'column "%" is generated or an identity column and cannot be derived', p_column;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_index WHERE indrelid = p_relid AND indisprimary AND v_target.attnum = ANY (indkey::int2[])) THEN
        RAISE EXCEPTION 'primary key column "%" cannot be derived', p_column;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_partitioned_table WHERE partrelid = p_relid AND v_target.attnum = ANY (partattrs::int2[])) THEN
        RAISE EXCEPTION 'partition key column "%" cannot be derived', p_column;
    END IF;
    IF EXISTS (SELECT 1 FROM eventual.derived_columns WHERE relid = p_relid AND attnum = v_target.attnum) THEN
        RAISE EXCEPTION 'column "%" of % is already derived', p_column, p_relid USING ERRCODE = 'duplicate_object';
    END IF;

    IF cardinality(p_inputs) <> cardinality(v_proc.input_types) THEN
        RAISE EXCEPTION 'processor "%" expects % input column(s) of type %, but derived column "%" lists %',
            p_processor, cardinality(v_proc.input_types), array_to_string(v_proc.input_types, ', '),
            p_column, cardinality(p_inputs);
    END IF;
    FOR i IN 1 .. cardinality(p_inputs) LOOP
        SELECT attnum, atttypid INTO v_att
        FROM pg_attribute WHERE attrelid = p_relid AND attname = p_inputs[i] AND attnum > 0 AND NOT attisdropped;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'column "%" used by derived column "%" does not exist in %', p_inputs[i], p_column, p_relid
                USING ERRCODE = 'undefined_column';
        END IF;
        IF v_att.attnum = v_target.attnum THEN
            RAISE EXCEPTION 'derived column "%" cannot depend on itself', p_column;
        END IF;
        IF NOT eventual._input_compatible(v_att.atttypid::regtype, v_proc.input_types[i], v_proc.kind) THEN
            RAISE EXCEPTION 'input column "%" has type %, but processor "%" expects % for input %',
                p_inputs[i], v_att.atttypid::regtype, p_processor, v_proc.input_types[i], i
                USING ERRCODE = 'datatype_mismatch';
        END IF;
        v_inputs := v_inputs || v_att.attnum;
    END LOOP;

    INSERT INTO eventual.derived_columns
        (relid, attnum, processor_id, input_attnums, target_name, input_names, params, effective_params)
    VALUES (p_relid, v_target.attnum, v_proc.id, v_inputs, p_column, p_inputs::name[], coalesce(p_params, '{}'),
            eventual._merge_params(v_proc.defaults, coalesce(p_params, '{}')))
    RETURNING id INTO v_id;
    RETURN v_id;
END
$$;

CREATE FUNCTION eventual._trigger_spec(p_relid regclass) RETURNS text
LANGUAGE sql STABLE AS $$
    SELECT jsonb_build_object(
               'v', 2,
               'pk', (SELECT to_jsonb(array_agg(k.attnum ORDER BY k.ord))
                        FROM pg_index i CROSS JOIN LATERAL unnest(i.indkey::int2[]) WITH ORDINALITY AS k(attnum, ord)
                       WHERE i.indrelid = p_relid AND i.indisprimary),
               'pk_names', (SELECT to_jsonb(array_agg(a.attname ORDER BY k.ord))
                              FROM pg_index i CROSS JOIN LATERAL unnest(i.indkey::int2[]) WITH ORDINALITY AS k(attnum, ord)
                              JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum
                             WHERE i.indrelid = p_relid AND i.indisprimary),
               'cols', coalesce((SELECT jsonb_agg(jsonb_build_object(
                                     'id', dc.id, 'att', dc.attnum, 'in', to_jsonb(dc.input_attnums), 'strict', p.strict,
                                     'name', dc.target_name, 'in_names', to_jsonb(dc.input_names))
                                     ORDER BY dc.depth, dc.id)
                                   FROM eventual.derived_columns dc JOIN eventual.processors p ON p.id = dc.processor_id
                                  WHERE dc.relid = p_relid), '[]'::jsonb))::text
$$;

CREATE FUNCTION eventual._sync_names(p_relid regclass) RETURNS void
LANGUAGE sql AS $$
    UPDATE eventual.derived_columns dc
    SET target_name = (SELECT attname FROM pg_attribute WHERE attrelid = dc.relid AND attnum = dc.attnum),
        input_names = (SELECT array_agg(a.attname ORDER BY u.ord)
                         FROM unnest(dc.input_attnums) WITH ORDINALITY AS u(attnum, ord)
                         JOIN pg_attribute a ON a.attrelid = dc.relid AND a.attnum = u.attnum)
    WHERE dc.relid = p_relid
$$;

CREATE FUNCTION eventual._repair_table(p_relid regclass) RETURNS integer
LANGUAGE plpgsql AS $$
DECLARE
    v_col record;
    v_target int2;
    v_inputs int2[];
    v_name name;
    v_att int2;
    v_changed integer := 0;
BEGIN
    FOR v_col IN SELECT * FROM eventual.derived_columns WHERE relid = p_relid ORDER BY id LOOP
        SELECT attnum INTO v_target FROM pg_attribute
        WHERE attrelid = p_relid AND attname = v_col.target_name AND attnum > 0 AND NOT attisdropped;
        IF v_target IS NULL THEN
            RAISE EXCEPTION 'cannot repair derived column binding %: column "%" is missing from %',
                v_col.id, v_col.target_name, p_relid;
        END IF;
        v_inputs := '{}';
        FOREACH v_name IN ARRAY v_col.input_names LOOP
            SELECT attnum INTO v_att FROM pg_attribute
            WHERE attrelid = p_relid AND attname = v_name AND attnum > 0 AND NOT attisdropped;
            IF v_att IS NULL THEN
                RAISE EXCEPTION 'cannot repair derived column binding %: input column "%" is missing from %',
                    v_col.id, v_name, p_relid;
            END IF;
            v_inputs := v_inputs || v_att;
        END LOOP;
        IF v_target <> v_col.attnum OR v_inputs <> v_col.input_attnums THEN
            UPDATE eventual.derived_columns SET attnum = v_target, input_attnums = v_inputs WHERE id = v_col.id;
            v_changed := v_changed + 1;
        END IF;
    END LOOP;
    RETURN v_changed;
END
$$;

CREATE FUNCTION eventual.repair() RETURNS integer
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $$
DECLARE
    v_table regclass;
    v_count integer := 0;
BEGIN
    FOR v_table IN SELECT DISTINCT relid FROM eventual.derived_columns LOOP
        PERFORM eventual._repair_table(v_table);
        PERFORM eventual._finalize_table(v_table);
        v_count := v_count + 1;
    END LOOP;
    RETURN v_count;
END
$$;

CREATE FUNCTION eventual._finalize_table(p_relid regclass) RETURNS void
LANGUAGE plpgsql AS $$
DECLARE
    v_cycle text;
    v_pk int2[];
    v_spec jsonb;
BEGIN
    WITH RECURSIVE cols AS (
        SELECT id, attnum, input_attnums FROM eventual.derived_columns WHERE relid = p_relid
    ), edges AS (
        SELECT child.id AS child, parent.id AS parent
        FROM cols child JOIN cols parent ON parent.attnum = ANY (child.input_attnums)
    ), walk (id, depth, path, is_cycle) AS (
        SELECT c.id, 0, ARRAY[c.id], false
        FROM cols c WHERE NOT EXISTS (SELECT 1 FROM edges e WHERE e.child = c.id)
        UNION ALL
        SELECT e.child, w.depth + 1, w.path || e.child, e.child = ANY (w.path)
        FROM walk w JOIN edges e ON e.parent = w.id
        WHERE NOT w.is_cycle
    ), problems AS (
        SELECT id FROM walk WHERE is_cycle
        UNION
        SELECT id FROM cols WHERE id NOT IN (SELECT id FROM walk)
    )
    SELECT string_agg(DISTINCT a.attname, ', ') INTO v_cycle
    FROM problems pr
    JOIN eventual.derived_columns dc ON dc.id = pr.id
    JOIN pg_attribute a ON a.attrelid = dc.relid AND a.attnum = dc.attnum;
    IF v_cycle IS NOT NULL THEN
        RAISE EXCEPTION 'derived columns of % form a dependency cycle involving: %', p_relid, v_cycle
            USING ERRCODE = 'invalid_object_definition';
    END IF;

    WITH RECURSIVE cols AS (
        SELECT id, attnum, input_attnums FROM eventual.derived_columns WHERE relid = p_relid
    ), edges AS (
        SELECT child.id AS child, parent.id AS parent
        FROM cols child JOIN cols parent ON parent.attnum = ANY (child.input_attnums)
    ), walk (id, depth) AS (
        SELECT c.id, 0 FROM cols c WHERE NOT EXISTS (SELECT 1 FROM edges e WHERE e.child = c.id)
        UNION ALL
        SELECT e.child, w.depth + 1 FROM walk w JOIN edges e ON e.parent = w.id
    )
    UPDATE eventual.derived_columns dc SET depth = d.depth
    FROM (SELECT id, max(depth) AS depth FROM walk GROUP BY id) d
    WHERE dc.id = d.id AND dc.depth IS DISTINCT FROM d.depth;

    SELECT array_agg(k.attnum ORDER BY k.ord) INTO v_pk
    FROM pg_index i CROSS JOIN LATERAL unnest(i.indkey::int2[]) WITH ORDINALITY AS k(attnum, ord)
    WHERE i.indrelid = p_relid AND i.indisprimary;

    v_spec := eventual._trigger_spec(p_relid)::jsonb;

    IF jsonb_array_length(v_spec -> 'cols') = 0 THEN
        EXECUTE format('DROP TRIGGER IF EXISTS zz_eventual_before ON %s', p_relid);
        EXECUTE format('DROP TRIGGER IF EXISTS zz_eventual_after ON %s', p_relid);
        EXECUTE format('DROP TRIGGER IF EXISTS zz_eventual_truncate ON %s', p_relid);
        EXECUTE format('DROP TRIGGER IF EXISTS zz_eventual_flush ON %s', p_relid);
        RETURN;
    END IF;
    IF v_pk IS NULL THEN
        RAISE EXCEPTION 'table % needs a primary key to have derived columns', p_relid;
    END IF;
    EXECUTE format('CREATE OR REPLACE TRIGGER zz_eventual_before BEFORE INSERT OR UPDATE ON %s '
                   'FOR EACH ROW EXECUTE FUNCTION eventual._row_trigger(%L)', p_relid, v_spec::text);
    EXECUTE format('CREATE OR REPLACE TRIGGER zz_eventual_after AFTER INSERT OR UPDATE OR DELETE ON %s '
                   'FOR EACH ROW EXECUTE FUNCTION eventual._row_trigger(%L)', p_relid, v_spec::text);
    EXECUTE format('CREATE OR REPLACE TRIGGER zz_eventual_flush AFTER INSERT OR UPDATE OR DELETE ON %s '
                   'FOR EACH STATEMENT EXECUTE FUNCTION eventual._flush_trigger()', p_relid);
    EXECUTE format('CREATE OR REPLACE TRIGGER zz_eventual_truncate AFTER TRUNCATE ON %s '
                   'FOR EACH STATEMENT EXECUTE FUNCTION eventual._truncate_trigger()', p_relid);
END
$$;

CREATE FUNCTION eventual._enqueue_all(p_column_id bigint) RETURNS bigint
LANGUAGE plpgsql AS $$
DECLARE
    v_col record;
    v_keys text;
    v_missing text;
    v_count bigint;
BEGIN
    SELECT dc.relid, dc.input_attnums, p.strict AS is_strict INTO v_col
    FROM eventual.derived_columns dc JOIN eventual.processors p ON p.id = dc.processor_id
    WHERE dc.id = p_column_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'derived column % does not exist', p_column_id;
    END IF;
    SELECT string_agg(format('%I::text', a.attname), ', ' ORDER BY k.ord) INTO v_keys
    FROM pg_index i
    CROSS JOIN LATERAL unnest(i.indkey::int2[]) WITH ORDINALITY AS k(attnum, ord)
    JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum
    WHERE i.indrelid = v_col.relid AND i.indisprimary;
    IF v_col.is_strict THEN
        SELECT string_agg(format('%I IS NULL', a.attname), ' OR ') INTO v_missing
        FROM unnest(v_col.input_attnums) AS u(attnum)
        JOIN pg_attribute a ON a.attrelid = v_col.relid AND a.attnum = u.attnum;
    ELSE
        v_missing := 'false';
    END IF;
    EXECUTE format($q$
        INSERT INTO eventual.cells AS c (column_id, row_key, state)
        SELECT %s, ARRAY[%s], CASE WHEN %s THEN 'waiting' ELSE 'pending' END FROM %s
        ON CONFLICT (column_id, row_key) DO UPDATE
        SET state = EXCLUDED.state, generation = c.generation + 1, attempts = 0,
            enqueued_at = now(), next_attempt_at = now(), started_at = NULL, finished_at = NULL,
            worker_pid = NULL, last_error = NULL, duration_ms = NULL, result_meta = NULL
        $q$, p_column_id, v_keys, v_missing, v_col.relid);
    GET DIAGNOSTICS v_count = ROW_COUNT;
    PERFORM eventual._wake_workers_on_commit();
    RETURN v_count;
END
$$;

CREATE FUNCTION eventual._after_column_type_change(p_relid regclass, p_columns name[]) RETURNS void
LANGUAGE plpgsql AS $$
DECLARE
    v_target text;
    v_col record;
    v_type regtype;
BEGIN
    SELECT string_agg(a.attname, ', ') INTO v_target
    FROM eventual.derived_columns dc
    JOIN pg_attribute a ON a.attrelid = dc.relid AND a.attnum = dc.attnum
    WHERE dc.relid = p_relid AND a.attname = ANY (p_columns);
    IF v_target IS NOT NULL THEN
        RAISE EXCEPTION 'cannot change the type of derived column(s) %', v_target
            USING HINT = 'Its type is the processor''s result type. Use eventual.drop_derived() first.',
                  ERRCODE = 'feature_not_supported';
    END IF;
    FOR v_col IN
        SELECT dc.id, dc.input_attnums, p.input_types, p.kind, p.name
        FROM eventual.derived_columns dc JOIN eventual.processors p ON p.id = dc.processor_id
        WHERE dc.relid = p_relid
          AND EXISTS (SELECT 1 FROM pg_attribute a
                      WHERE a.attrelid = p_relid AND a.attnum = ANY (dc.input_attnums) AND a.attname = ANY (p_columns))
    LOOP
        FOR i IN 1 .. cardinality(v_col.input_attnums) LOOP
            SELECT atttypid::regtype INTO v_type FROM pg_attribute
            WHERE attrelid = p_relid AND attnum = v_col.input_attnums[i];
            IF NOT eventual._input_compatible(v_type, v_col.input_types[i], v_col.kind) THEN
                RAISE EXCEPTION 'processor "%" expects % for input %, but the column would become %',
                    v_col.name, v_col.input_types[i], i, v_type USING ERRCODE = 'datatype_mismatch';
            END IF;
        END LOOP;
        PERFORM eventual._enqueue_all(v_col.id);
    END LOOP;
END
$$;

CREATE FUNCTION eventual.add_derived_column(
    tbl regclass, column_name name, processor text, inputs text[], params jsonb DEFAULT '{}'
) RETURNS bigint
LANGUAGE plpgsql AS $$
DECLARE
    v_id bigint;
BEGIN
    IF coalesce(cardinality(inputs), 0) = 0 THEN
        RAISE EXCEPTION 'add_derived_column needs at least one input column';
    END IF;
    EXECUTE format('ALTER TABLE %s ADD COLUMN %I eventual.derived(%L, %s, %L)',
                   tbl, column_name, processor,
                   (SELECT string_agg(quote_literal(i), ', ' ORDER BY o)
                      FROM unnest(inputs) WITH ORDINALITY AS u(i, o)),
                   coalesce(params, '{}')::text);
    SELECT cs.column_id INTO v_id
    FROM eventual.column_status cs
    WHERE cs.table_name = tbl AND cs.column_name = add_derived_column.column_name;
    RETURN v_id;
END
$$;

CREATE FUNCTION eventual._drop_derived(tbl regclass, column_name name, unused boolean) RETURNS bigint
LANGUAGE plpgsql AS $$
DECLARE
    v_col record;
    v_dependents text;
BEGIN
    SELECT dc.id, dc.attnum INTO v_col
    FROM eventual.derived_columns dc
    JOIN pg_attribute a ON a.attrelid = dc.relid AND a.attnum = dc.attnum
    WHERE dc.relid = tbl AND a.attname = column_name;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'column "%" of % is not a derived column', column_name, tbl;
    END IF;
    SELECT string_agg(a.attname, ', ') INTO v_dependents
    FROM eventual.derived_columns dc
    JOIN pg_attribute a ON a.attrelid = dc.relid AND a.attnum = dc.attnum
    WHERE dc.relid = tbl AND v_col.attnum = ANY (dc.input_attnums);
    IF v_dependents IS NOT NULL THEN
        RAISE EXCEPTION 'derived column(s) % depend on "%"; drop them first', v_dependents, column_name
            USING ERRCODE = 'dependent_objects_still_exist';
    END IF;
    DELETE FROM eventual.derived_columns WHERE id = v_col.id;
    PERFORM eventual._finalize_table(tbl);
    RETURN 1;
END
$$;

CREATE FUNCTION eventual._refresh(tbl regclass, column_name name) RETURNS bigint
LANGUAGE plpgsql AS $$
DECLARE
    v_total bigint := 0;
    v_id bigint;
    v_found boolean := false;
BEGIN
    FOR v_id IN
        SELECT dc.id
        FROM eventual.derived_columns dc
        JOIN pg_attribute a ON a.attrelid = dc.relid AND a.attnum = dc.attnum
        WHERE dc.relid = tbl AND (column_name IS NULL OR a.attname = column_name)
        ORDER BY dc.depth, dc.id
    LOOP
        v_found := true;
        v_total := v_total + eventual._enqueue_all(v_id);
    END LOOP;
    IF NOT v_found AND column_name IS NOT NULL THEN
        RAISE EXCEPTION 'column "%" of % is not a derived column', column_name, tbl;
    END IF;
    RETURN v_total;
END
$$;

CREATE FUNCTION eventual._retry_failed(tbl regclass, column_name name) RETURNS bigint
LANGUAGE plpgsql AS $$
DECLARE
    v_count bigint;
BEGIN
    UPDATE eventual.cells c
    SET state = 'pending', attempts = 0, generation = c.generation + 1, next_attempt_at = now(),
        enqueued_at = now(), last_error = NULL, started_at = NULL, finished_at = NULL, worker_pid = NULL
    FROM eventual.derived_columns dc
    JOIN pg_attribute a ON a.attrelid = dc.relid AND a.attnum = dc.attnum
    WHERE c.column_id = dc.id AND c.state = 'failed' AND dc.relid = tbl
      AND (column_name IS NULL OR a.attname = column_name);
    GET DIAGNOSTICS v_count = ROW_COUNT;
    PERFORM eventual._wake_workers_on_commit();
    RETURN v_count;
END
$$;

CREATE FUNCTION eventual.refresh(tbl regclass, column_name name DEFAULT NULL) RETURNS bigint
LANGUAGE sql AS $$ SELECT eventual._table_admin('refresh', tbl::oid, column_name::text, false) $$;

CREATE FUNCTION eventual.retry_failed(tbl regclass, column_name name DEFAULT NULL) RETURNS bigint
LANGUAGE sql AS $$ SELECT eventual._table_admin('retry_failed', tbl::oid, column_name::text, false) $$;

CREATE FUNCTION eventual.drop_derived(tbl regclass, column_name name, drop_column boolean DEFAULT false)
RETURNS void
LANGUAGE plpgsql AS $$
BEGIN
    PERFORM eventual._table_admin('drop_derived', tbl::oid, column_name::text, drop_column);
    IF drop_column THEN
        EXECUTE format('ALTER TABLE %s DROP COLUMN %I', tbl, column_name);
    END IF;
END
$$;

CREATE FUNCTION eventual.set_secret(name text, value text) RETURNS void
LANGUAGE sql SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $$
    INSERT INTO eventual.secrets (name, value) VALUES (name, value)
    ON CONFLICT (name) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
$$;

CREATE FUNCTION eventual.drop_secret(name text) RETURNS boolean
LANGUAGE sql SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $$
    WITH d AS (DELETE FROM eventual.secrets s WHERE s.name = drop_secret.name RETURNING 1)
    SELECT EXISTS (SELECT 1 FROM d)
$$;

CREATE FUNCTION eventual.secret_names() RETURNS SETOF text
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $$
    SELECT name FROM eventual.secrets ORDER BY name
$$;

CREATE FUNCTION eventual.pause() RETURNS void
LANGUAGE sql SECURITY DEFINER SET search_path = pg_catalog, pg_temp
AS $$ UPDATE eventual.control SET paused = true, updated_at = now() $$;

CREATE FUNCTION eventual.resume() RETURNS void
LANGUAGE sql SECURITY DEFINER SET search_path = pg_catalog, pg_temp
AS $$ UPDATE eventual.control SET paused = false, updated_at = now();
      SELECT eventual._wake_workers_on_commit() $$;

CREATE FUNCTION eventual.wait(tbl regclass DEFAULT NULL, timeout interval DEFAULT '60 seconds')
RETURNS boolean
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $$
DECLARE
    v_deadline float8 := eventual._monotonic_seconds() + extract(epoch FROM timeout);
BEGIN
    LOOP
        IF NOT EXISTS (
            SELECT 1 FROM eventual.cells c JOIN eventual.derived_columns dc ON dc.id = c.column_id
            WHERE c.state IN ('pending', 'running') AND (tbl IS NULL OR dc.relid = tbl)) THEN
            RETURN true;
        END IF;
        IF eventual._monotonic_seconds() >= v_deadline THEN
            RETURN false;
        END IF;
        PERFORM eventual._nap(20);
    END LOOP;
END
$$;

CREATE FUNCTION eventual.wait_for_workers(paused boolean, timeout interval DEFAULT '10 seconds')
RETURNS boolean
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $$
DECLARE
    v_deadline float8 := eventual._monotonic_seconds() + extract(epoch FROM timeout);
    v_expected text := CASE WHEN paused THEN 'paused' ELSE 'running' END;
BEGIN
    LOOP
        PERFORM pg_stat_clear_snapshot();
        IF EXISTS (SELECT 1 FROM pg_stat_activity
                   WHERE backend_type = 'pg_eventual worker' AND datname = current_database())
           AND NOT EXISTS (
               SELECT 1 FROM pg_stat_activity sa
               LEFT JOIN eventual.workers w ON w.pid = sa.pid
               WHERE sa.backend_type = 'pg_eventual worker' AND sa.datname = current_database()
                 AND w.state IS DISTINCT FROM v_expected) THEN
            RETURN true;
        END IF;
        IF eventual._monotonic_seconds() >= v_deadline THEN
            RETURN false;
        END IF;
        PERFORM eventual._nap(20);
    END LOOP;
END
$$;

CREATE VIEW eventual.column_status AS
SELECT dc.id AS column_id,
       dc.relid AS table_name,
       a.attname AS column_name,
       p.name AS processor,
       p.kind,
       (SELECT array_agg(ia.attname ORDER BY u.ord)
          FROM unnest(dc.input_attnums) WITH ORDINALITY AS u(attnum, ord)
          JOIN pg_attribute ia ON ia.attrelid = dc.relid AND ia.attnum = u.attnum) AS inputs,
       dc.effective_params,
       dc.depth,
       count(c.*) FILTER (WHERE c.state = 'waiting') AS waiting,
       count(c.*) FILTER (WHERE c.state = 'pending') AS pending,
       count(c.*) FILTER (WHERE c.state = 'running') AS running,
       count(c.*) FILTER (WHERE c.state = 'done') AS done,
       count(c.*) FILTER (WHERE c.state = 'failed') AS failed
FROM eventual.derived_columns dc
JOIN eventual.processors p ON p.id = dc.processor_id
JOIN pg_attribute a ON a.attrelid = dc.relid AND a.attnum = dc.attnum
LEFT JOIN eventual.cells c ON c.column_id = dc.id
WHERE has_table_privilege(dc.relid, 'SELECT')
GROUP BY dc.id, a.attname, p.name, p.kind;

CREATE VIEW eventual.cell_status AS
SELECT dc.relid AS table_name,
       a.attname AS column_name,
       c.row_key,
       c.state,
       c.attempts,
       c.generation,
       p.name AS processor,
       c.enqueued_at,
       c.started_at,
       c.finished_at,
       c.duration_ms,
       c.last_error,
       c.result_meta
FROM eventual.cells c
JOIN eventual.derived_columns dc ON dc.id = c.column_id
JOIN eventual.processors p ON p.id = dc.processor_id
JOIN pg_attribute a ON a.attrelid = dc.relid AND a.attnum = dc.attnum
WHERE has_table_privilege(dc.relid, 'SELECT');

CREATE VIEW eventual._column_specs AS
SELECT dc.id AS column_id,
       format('%I.%I', n.nspname, c.relname) AS table_sql,
       c.relowner::bigint AS owner_oid,
       quote_ident(ta.attname) AS target_sql,
       ta.attname::text AS target_name,
       (SELECT array_agg(quote_ident(ia.attname) ORDER BY u.ord)
          FROM unnest(dc.input_attnums) WITH ORDINALITY AS u(attnum, ord)
          JOIN pg_attribute ia ON ia.attrelid = dc.relid AND ia.attnum = u.attnum) AS inputs_sql,
       (SELECT array_agg(quote_ident(ka.attname) ORDER BY k.ord)
          FROM pg_index i
          CROSS JOIN LATERAL unnest(i.indkey::int2[]) WITH ORDINALITY AS k(attnum, ord)
          JOIN pg_attribute ka ON ka.attrelid = i.indrelid AND ka.attnum = k.attnum
         WHERE i.indrelid = dc.relid AND i.indisprimary) AS pk_sql,
       (SELECT array_agg(format_type(ka.atttypid, ka.atttypmod) ORDER BY k.ord)
          FROM pg_index i
          CROSS JOIN LATERAL unnest(i.indkey::int2[]) WITH ORDINALITY AS k(attnum, ord)
          JOIN pg_attribute ka ON ka.attrelid = i.indrelid AND ka.attnum = k.attnum
         WHERE i.indrelid = dc.relid AND i.indisprimary) AS pk_types,
       p.name AS processor,
       p.version AS processor_version,
       p.kind,
       (SELECT format('%I.%I', pn.nspname, pp.proname)
          FROM pg_proc pp JOIN pg_namespace pn ON pn.oid = pp.pronamespace
         WHERE pp.oid = p.function) AS fn_sql,
       (SELECT array_agg(CASE WHEN t.typtype = 'p' THEN NULL ELSE format_type(t.oid, NULL) END ORDER BY u.ord)
          FROM unnest(p.input_types) WITH ORDINALITY AS u(typ, ord)
          JOIN pg_type t ON t.oid = u.typ) AS input_types,
       p.defaults <> '{}'::jsonb AS takes_params,
       p.config,
       dc.effective_params,
       p.strict,
       p.max_attempts,
       (SELECT t.typcategory = 'A' FROM pg_type t WHERE t.oid = p.result_type) AS result_is_array
FROM eventual.derived_columns dc
JOIN pg_class c ON c.oid = dc.relid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_attribute ta ON ta.attrelid = dc.relid AND ta.attnum = dc.attnum
JOIN eventual.processors p ON p.id = dc.processor_id;

CREATE FUNCTION eventual._on_sql_drop() RETURNS event_trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $$
DECLARE
    v_relids oid[];
    v_attnums int2[];
    v_problem text;
    v_table oid;
BEGIN
    IF to_regclass('eventual.derived_columns') IS NULL THEN
        RETURN;
    END IF;
    SELECT array_agg(objid), array_agg(objsubid::int2) INTO v_relids, v_attnums
    FROM pg_event_trigger_dropped_objects()
    WHERE classid = 'pg_class'::regclass AND object_type IN ('table', 'table column');
    IF v_relids IS NULL OR NOT EXISTS (
        SELECT 1 FROM eventual.derived_columns WHERE relid::oid = ANY (v_relids)) THEN
        RETURN;
    END IF;

    DELETE FROM eventual.derived_columns dc
    USING unnest(v_relids, v_attnums) AS d(relid, attnum)
    WHERE d.attnum = 0 AND dc.relid::oid = d.relid;

    SELECT string_agg(format('%s.%I (input of %I)', dc.relid,
                             dc.input_names[array_position(dc.input_attnums, d.attnum)], dc.target_name), ', ')
    INTO v_problem
    FROM eventual.derived_columns dc
    JOIN unnest(v_relids, v_attnums) AS d(relid, attnum)
      ON d.relid = dc.relid::oid AND d.attnum > 0 AND d.attnum = ANY (dc.input_attnums)
    WHERE NOT EXISTS (
        SELECT 1 FROM unnest(v_relids, v_attnums) AS t(relid, attnum)
        WHERE t.relid = dc.relid::oid AND t.attnum = dc.attnum);
    IF v_problem IS NOT NULL THEN
        RAISE EXCEPTION 'cannot drop column(s) used by derived columns: %', v_problem
            USING HINT = 'Drop the derived columns first (or together), or use eventual.drop_derived().',
                  ERRCODE = 'dependent_objects_still_exist';
    END IF;

    DELETE FROM eventual.derived_columns dc
    USING unnest(v_relids, v_attnums) AS d(relid, attnum)
    WHERE d.attnum > 0 AND dc.relid::oid = d.relid AND dc.attnum = d.attnum;
    FOR v_table IN
        SELECT DISTINCT d.relid FROM unnest(v_relids, v_attnums) AS d(relid, attnum)
        WHERE d.attnum > 0 AND EXISTS (SELECT 1 FROM pg_class WHERE oid = d.relid)
    LOOP
        PERFORM eventual._finalize_table(v_table::regclass);
    END LOOP;
END
$$;

CREATE EVENT TRIGGER eventual_sql_drop ON sql_drop EXECUTE FUNCTION eventual._on_sql_drop();

GRANT USAGE ON SCHEMA eventual TO PUBLIC;
GRANT SELECT ON eventual.processors, eventual.column_status, eventual.cell_status,
                eventual.workers, eventual.control TO PUBLIC;
REVOKE EXECUTE ON FUNCTION
    eventual.create_processor(text, text, regprocedure, regtype[], regtype, jsonb, jsonb, boolean, integer, text),
    eventual.drop_processor(text),
    eventual.alter_processor(text, jsonb, jsonb, regprocedure, boolean, integer, text, boolean),
    eventual.set_secret(text, text),
    eventual.drop_secret(text),
    eventual.secret_names(),
    eventual.pause(),
    eventual.resume(),
    eventual._bind_column(regclass, name, text, text[], jsonb),
    eventual._finalize_table(regclass),
    eventual._enqueue_all(bigint),
    eventual._drop_derived(regclass, name, boolean),
    eventual._refresh(regclass, name),
    eventual._retry_failed(regclass, name),
    eventual._after_column_type_change(regclass, name[]),
    eventual._sync_names(regclass),
    eventual._repair_table(regclass),
    eventual.repair()
FROM PUBLIC;

SELECT pg_catalog.pg_extension_config_dump('eventual.processors', '');
SELECT pg_catalog.pg_extension_config_dump('eventual.derived_columns', '');
SELECT pg_catalog.pg_extension_config_dump('eventual.cells', '');
SELECT pg_catalog.pg_extension_config_dump('eventual.processors_id_seq', '');
SELECT pg_catalog.pg_extension_config_dump('eventual.derived_columns_id_seq', '');
