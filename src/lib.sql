CREATE SCHEMA neon_ai_schema;

CREATE TABLE neon_ai_schema.config(_name text PRIMARY KEY, _value text);

REVOKE ALL ON TABLE neon_ai_schema.config FROM PUBLIC;

CREATE FUNCTION set_openai_api_key(_key text) RETURNS void
LANGUAGE SQL VOLATILE STRICT PARALLEL UNSAFE AS $$
    INSERT INTO neon_ai_schema.config VALUES ('OPENAI_KEY', _key)
    ON CONFLICT (_name) DO UPDATE SET _value = EXCLUDED._value;
$$;

CREATE FUNCTION get_openai_api_key() RETURNS text
LANGUAGE SQL VOLATILE STRICT PARALLEL SAFE AS $$
    SELECT _value FROM neon_ai_schema.config WHERE _name = 'OPENAI_KEY';
$$;

CREATE FUNCTION embedding_openai(_model text, _input text) RETURNS real[]
LANGUAGE PLPGSQL IMMUTABLE STRICT PARALLEL UNSAFE AS $$
    DECLARE
        _key text := get_openai_api_key();
        _res real[];
    BEGIN
        IF _key IS NULL THEN
            RAISE EXCEPTION '[NEON_AI] OpenAI API key is NULL or has not been set';
        END IF;
        SELECT array_agg(_tmp._js::real) FROM (
            SELECT jsonb_path_query(
                embedding_openai_raw(_model, _input, _key),
                '$.data.embedding.double()'
            ) AS _js
        ) AS _tmp INTO _res;
        RETURN _res;
    END;
$$;
