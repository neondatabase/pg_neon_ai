CREATE SCHEMA neon_ai_schema;

CREATE TABLE neon_ai_schema.config(name text PRIMARY KEY, value text);

REVOKE ALL ON TABLE neon_ai_schema.config FROM PUBLIC;

CREATE FUNCTION set_openai_api_key(api_key text) RETURNS void
LANGUAGE SQL VOLATILE STRICT PARALLEL UNSAFE AS $$
    INSERT INTO neon_ai_schema.config VALUES ('OPENAI_KEY', api_key)
    ON CONFLICT (name) DO UPDATE SET value = EXCLUDED.value;
$$;

CREATE FUNCTION get_openai_api_key() RETURNS text
LANGUAGE SQL VOLATILE STRICT PARALLEL SAFE AS $$
    SELECT value FROM neon_ai_schema.config WHERE name = 'OPENAI_KEY';
$$;

CREATE FUNCTION embedding_openai(model text, input text) RETURNS real[]
LANGUAGE PLPGSQL IMMUTABLE STRICT PARALLEL UNSAFE AS $$
    DECLARE
        api_key text := get_openai_api_key();
        res real[];
    BEGIN
        IF api_key IS NULL THEN
            RAISE EXCEPTION '[NEON_AI] OpenAI API key is NULL or has not been set';
        END IF;
        SELECT array_agg(tmp.js::real) FROM (
            SELECT jsonb_path_query(
                embedding_openai_raw(model, input, api_key),
                '$.data.embedding.double()'
            ) AS js
        ) AS tmp INTO res;
        RETURN res;
    END;
$$;


CREATE FUNCTION chatgpt(body json) RETURNS json
LANGUAGE PLPGSQL IMMUTABLE STRICT PARALLEL UNSAFE AS $$
    DECLARE
        api_key text := get_openai_api_key();
        res jsonb;
    BEGIN
        IF api_key IS NULL THEN
            RAISE EXCEPTION '[NEON_AI] OpenAI API key is NULL or has not been set';
        END IF;
        SELECT chatgpt_raw(body, api_key) INTO res;
        RETURN res;
    END;
$$;

CREATE FUNCTION embedding_bge_small_en_v15(input text) RETURNS vector(384) 
LANGUAGE SQL VOLATILE STRICT PARALLEL SAFE AS $$
    SELECT embedding_bge_small_en_v15_raw(input)::vector(384);
$$;
