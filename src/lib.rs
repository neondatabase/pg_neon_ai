use fastembed::{TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel};
use pgrx::prelude::*;
use std::cell::OnceCell;

pgrx::pg_module_magic!();

extension_sql_file!("lib.sql");

const ERR_PREFIX: &'static str = "[NEON_AI]";

#[pg_extern(immutable, strict)]
fn embedding_openai_raw(model: &str, input: &str, key: &str) -> pgrx::JsonB {
    let auth = format!("Bearer {key}");
    let json_body = ureq::json!({ "model": model, "input": input });

    match ureq::post("https://api.openai.com/v1/embeddings")
        .set("Authorization", auth.as_str())
        .send_json(json_body)
    {
        Err(ureq::Error::Transport(err)) => {
            let msg = err.message().unwrap_or("no further details");
            error!("{ERR_PREFIX} Transport error communicating with OpenAI API: {msg}");
        }
        Err(ureq::Error::Status(code, _)) => error!("{ERR_PREFIX} HTTP status code {code} trying to reach OpenAI API"),
        Ok(response) => match response.into_json() {
            Err(err) => error!("{ERR_PREFIX} Failed to parse JSON received from OpenAI API: {err}"),
            Ok(value) => pgrx::JsonB(value),
        },
    }
}

#[pg_extern]
fn test_nested() -> Vec<Vec<f32>> {
    vec![vec![1.1, 1.2, 1.3], vec![2.1, 2.2, 2.3]]
}

#[pg_extern(immutable, strict, name = "embedding_bge_small_en_v15")]
fn embeddings_bge_small_en_v15(input: Vec<&str>) -> Vec<Vec<f32>> {
    thread_local! {
        static MODEL_CELL: OnceCell<TextEmbedding> = const { OnceCell::new() };
    }
    MODEL_CELL.with(|cell| {
        let model = cell.get_or_init(|| {
            let user_def_model = UserDefinedEmbeddingModel {
                onnx_file: include_bytes!("../bge_small_en_v15/model.onnx").to_vec(),
                tokenizer_files: TokenizerFiles {
                    tokenizer_file: include_bytes!("../bge_small_en_v15/tokenizer.json").to_vec(),
                    config_file: include_bytes!("../bge_small_en_v15/config.json").to_vec(),
                    special_tokens_map_file: include_bytes!("../bge_small_en_v15/special_tokens_map.json").to_vec(),
                    tokenizer_config_file: include_bytes!("../bge_small_en_v15/tokenizer_config.json").to_vec(),
                },
            };
            match TextEmbedding::try_new_from_user_defined(user_def_model, Default::default()) {
                Err(err) => error!("{ERR_PREFIX} Couldn't load model bge_small_en_v15: {err}"),
                Ok(result) => result,
            }
        });
        match model.embed(input, None) {
            Err(err) => error!("{ERR_PREFIX} Unable to generate bge_small_en_v15 embeddings: {err}"),
            Ok(vectors) => vectors,
        }
    })
}

#[pg_extern(immutable, strict, name = "embedding_bge_small_en_v15")]
fn embedding_bge_small_en_v15(input: &str) -> Vec<f32> {
    let vectors = embeddings_bge_small_en_v15(vec![input]);
    match vectors.into_iter().next() {
        None => error!("{ERR_PREFIX} Unexpected empty result vector"),
        Some(vector) => vector,
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;
    use serde_json;
    use std::env;

    fn get_openai_api_key() -> String {
        match env::var("OPENAI_API_KEY") {
            Err(err) => error!("Tests require environment variable OPENAI_API_KEY: {}", err),
            Ok(key) => key,
        }
    }

    #[pg_test(error = "[NEON_AI] HTTP status code 401 trying to reach OpenAI API")]
    fn test_embedding_openai_raw_bad_key() {
        crate::embedding_openai_raw("text-embedding-3-small", "hello world!", "invalid-key");
    }

    #[pg_test(error = "[NEON_AI] HTTP status code 404 trying to reach OpenAI API")]
    fn test_embedding_openai_raw_bad_model() {
        crate::embedding_openai_raw("text-embedding-3-immense", "hello world!", &get_openai_api_key());
    }

    #[pg_test]
    fn test_embedding_openai_raw_has_data() {
        let json = crate::embedding_openai_raw("text-embedding-3-small", "hello world!", &get_openai_api_key());
        let result = match json.0 {
            serde_json::Value::Object(obj) => obj,
            _ => error!("Unexpected non-Object JSON type"),
        };
        assert!(result.contains_key("data"));
    }

    #[pg_test]
    fn test_embedding_bge_small_en_v15_length() {
        assert!(crate::embedding_bge_small_en_v15("hello world!").len() == 384);
    }

    #[pg_test]
    fn test_embedding_bge_small_en_v15_immutability() {
        assert!(crate::embedding_bge_small_en_v15("hello world!") == crate::embedding_bge_small_en_v15("hello world!"));
    }

    #[pg_test]
    fn test_embedding_bge_small_en_v15_variability() {
        assert!(crate::embedding_bge_small_en_v15("hello world!") != crate::embedding_bge_small_en_v15("bye moon!"));
    }
}

/// This module is required by `cargo pgrx test` invocations.
/// It must be visible at the root of your extension crate.
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {
        // perform one-off initialization when the pg_test framework starts
    }

    pub fn postgresql_conf_options() -> Vec<&'static str> {
        // return any postgresql.conf settings that are required for your tests
        vec![]
    }
}
