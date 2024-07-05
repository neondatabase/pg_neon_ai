// use fastembed::{EmbeddingModel, InitOptions, TokenizerFiles}
use fastembed::{TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel};
use pgrx::prelude::*;
use std::cell::OnceCell;

pgrx::pg_module_magic!();

extension_sql_file!("lib.sql");

const ERR_PREFIX: &'static str = "[NEON_AI]";

#[pg_extern]
fn hello_neon_ai() -> &'static str {
    "Hello, neon_ai"
}

#[pg_extern]
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
        Err(ureq::Error::Status(code, _)) => {
            // unexpected status code (such as 400, 500 etc)
            error!("{ERR_PREFIX} HTTP status code {code} trying to reach OpenAI API");
        }
        Ok(response) => match response.into_json() {
            Err(err) => {
                error!("{ERR_PREFIX} Failed to parse JSON received from OpenAI API: {err}");
            }
            Ok(value) => pgrx::JsonB(value),
        },
    }
}

#[pg_extern]
fn embedding_bge_small_en_v15(input: Vec<&str>) -> Vec<Vec<f32>> {
    thread_local! {
        static MODEL_CELL: OnceCell<TextEmbedding> = const { OnceCell::new() };
    }
    MODEL_CELL.with(|cell| {
        let model = cell.get_or_init(|| {
            let onnx_file = include_bytes!("../bge_small_en_v15/model.onnx").to_vec();
            let tokenizer_file = include_bytes!("../bge_small_en_v15/tokenizer.json").to_vec();
            let config_file = include_bytes!("../bge_small_en_v15/config.json").to_vec();
            let special_tokens_map_file = include_bytes!("../bge_small_en_v15/special_tokens_map.json").to_vec();
            let tokenizer_config_file = include_bytes!("../bge_small_en_v15/tokenizer_config.json").to_vec();

            let user_def_model = UserDefinedEmbeddingModel {
                onnx_file,
                tokenizer_files: TokenizerFiles {
                    tokenizer_file,
                    config_file,
                    special_tokens_map_file,
                    tokenizer_config_file,
                },
            };

            match TextEmbedding::try_new_from_user_defined(user_def_model, Default::default()) {
                Err(err) => {
                    error!("{ERR_PREFIX} Couldn't load model bge_small_en_v15: {err}");
                }
                Ok(result) => result,
            }
        });

        match model.embed(input, None) {
            Err(err) => {
                error!("{ERR_PREFIX} Unable to generate bge_small_en_v15 embeddings: {err}",);
            }
            Ok(result) => result,
        }
    })
}

// this downloads the model on-the-fly
// let model = match TextEmbedding::try_new(InitOptions {
//     model_name: EmbeddingModel::BGESmallENV15,
//     show_download_progress: false,
//     ..Default::default()
// }) {

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_hello_neon_ai() {
        assert_eq!("Hello, neon_ai", crate::hello_neon_ai());
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
