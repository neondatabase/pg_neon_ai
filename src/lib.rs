use fastembed::{
    RerankResult, TextEmbedding, TextRerank, TokenizerFiles, UserDefinedEmbeddingModel, UserDefinedRerankingModel,
};
use pgrx::prelude::*;
use std::cell::OnceCell;
use text_splitter::{ChunkConfig, TextSplitter};
use tokenizers::{AddedToken, Tokenizer};

pgrx::pg_module_magic!();
extension_sql_file!("lib.sql");
const ERR_PREFIX: &'static str = "[NEON_AI]";

// NOTE: these macros assume /unix/style/paths
macro_rules! local_tokenizer_files {
    ($folder:literal) => {
        TokenizerFiles {
            tokenizer_file: include_bytes!(concat!($folder, "/tokenizer.json")).to_vec(),
            config_file: include_bytes!(concat!($folder, "/config.json")).to_vec(),
            special_tokens_map_file: include_bytes!(concat!($folder, "/special_tokens_map.json")).to_vec(),
            tokenizer_config_file: include_bytes!(concat!($folder, "/tokenizer_config.json")).to_vec(),
        }
    };
}

macro_rules! local_model {
    ($model:ident, $folder:literal) => {
        $model {
            onnx_file: include_bytes!(concat!($folder, "/model.onnx")).to_vec(),
            tokenizer_files: local_tokenizer_files!($folder),
        }
    };
}

// === OpenAI embeddings ===

#[pg_extern(immutable, strict)]
fn embedding_openai_raw(model: &str, input: &str, key: &str) -> pgrx::JsonB {
    let auth = format!("Bearer {key}");
    let json_body = ureq::json!({ "model": model, "input": input });

    let response = match ureq::post("https://api.openai.com/v1/embeddings")
        .set("Authorization", auth.as_str())
        .send_json(json_body)
    {
        Err(ureq::Error::Transport(err)) => {
            let msg = err.message().unwrap_or("no further details");
            error!("{ERR_PREFIX} Transport error communicating with OpenAI API: {msg}");
        }
        Err(ureq::Error::Status(code, _)) => {
            error!("{ERR_PREFIX} HTTP status code {code} trying to reach OpenAI API")
        }
        Ok(response) => response,
    };
    match response.into_json() {
        Err(err) => error!("{ERR_PREFIX} Failed to parse JSON received from OpenAI API: {err}"),
        Ok(value) => pgrx::JsonB(value),
    }
}

// === Local embeddings ===

// NOTE. It might be nice to expose this function directly, but as at 2024-07-08 pgrx
// doesn't support Vec<Vec<_>>: https://github.com/pgcentralfoundation/pgrx/issues/1762.
// #[pg_extern(immutable, strict, name = "embedding_bge_small_en_v15")]
fn embeddings_bge_small_en_v15(input: Vec<&str>) -> Vec<Vec<f32>> {
    thread_local! {
        static CELL: OnceCell<TextEmbedding> = const { OnceCell::new() };
    }
    CELL.with(|cell| {
        let model = cell.get_or_init(|| {
            let user_def_model = local_model!(UserDefinedEmbeddingModel, "../bge_small_en_v15");
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

#[pg_extern(immutable, strict)]
fn embedding_bge_small_en_v15(input: &str) -> Vec<f32> {
    let vectors = embeddings_bge_small_en_v15(vec![input]);
    match vectors.into_iter().next() {
        None => error!("{ERR_PREFIX} Unexpected empty result vector"),
        Some(vector) => vector,
    }
}

// === Local reranking ===

fn reranks_jina_v1_tiny_en_base(query: &str, documents: Vec<&str>) -> Vec<RerankResult> {
    thread_local! {
        static CELL: OnceCell<TextRerank> = const { OnceCell::new() };
    }
    CELL.with(|cell| {
        let model = cell.get_or_init(|| {
            let user_def_model = local_model!(UserDefinedRerankingModel, "../jina_reranker_v1_tiny_en");
            match TextRerank::try_new_from_user_defined(user_def_model, Default::default()) {
                Err(err) => error!("{ERR_PREFIX} Couldn't load model jina_reranker_v1_turbo_en: {err}"),
                Ok(result) => result,
            }
        });
        match model.rerank(query, documents, false, None) {
            Err(err) => error!("{ERR_PREFIX} Unable to rerank with jina_reranker_v1_turbo_en: {err}"),
            Ok(rr) => rr,
        }
    })
}

#[pg_extern(immutable, strict)]
fn rerank_indices_jina_v1_tiny_en(query: &str, documents: Vec<&str>) -> Vec<i32> {
    let reranking = reranks_jina_v1_tiny_en_base(query, documents);
    reranking.iter().map(|rr| rr.index as i32).collect()
}

#[pg_extern(immutable, strict)]
fn rerank_scores_jina_v1_tiny_en(query: &str, documents: Vec<&str>) -> Vec<f32> {
    let mut reranking = reranks_jina_v1_tiny_en_base(query, documents);
    reranking.sort_by(|rr1, rr2| rr1.index.cmp(&rr2.index)); // return to input order
    reranking.iter().map(|rr| rr.score as f32).collect()
}

#[pg_extern(immutable, strict)]
fn rerank_score_jina_v1_tiny_en(query: &str, document: &str) -> f32 {
    let scores = rerank_scores_jina_v1_tiny_en(query, vec![document]);
    match scores.first() {
        None => error!("{ERR_PREFIX} Unexpectedly empty reranking vector"),
        Some(score) => *score,
    }
}

#[pg_extern(immutable, strict)]
fn chunks_by_characters(document: &str, max_characters: i32, max_overlap: i32) -> Vec<&str> {
    if max_characters < 1 || max_overlap < 0 {
        error!("{ERR_PREFIX} max_characters must be >= 1 and max_overlap must be >= 0");
    }
    let config = match ChunkConfig::new(max_characters as usize).with_overlap(max_overlap as usize) {
        Err(err) => error!("{ERR_PREFIX} Error creating chunk config: {err}"),
        Ok(config) => config,
    };
    let splitter = TextSplitter::new(config);
    let chunks = splitter.chunks(document).collect();
    chunks
}

#[pg_extern(immutable, strict)]
fn chunks_by_tokens(document: &str, max_tokens: i32, max_overlap: i32) -> Vec<&str> {
    thread_local! {
        static CELL: OnceCell<(Tokenizer, i32)> = const { OnceCell::new() };
    }
    CELL.with(|cell| {
        let (tokenizer, model_max_length) = cell.get_or_init(|| {
            let mut tokenizer = match Tokenizer::from_bytes(include_bytes!("../bge_small_en_v15/tokenizer.json")) {
                Err(err) => error!("{ERR_PREFIX} Error loading tokenizer: {err}"),
                Ok(tokenizer) => tokenizer,
            };
            let special_tokens_map: serde_json::Value =
                match serde_json::from_slice(include_bytes!("../bge_small_en_v15/special_tokens_map.json")) {
                    Err(err) => error!("{ERR_PREFIX} Error loading special tokens: {err}"),
                    Ok(map) => map,
                };
            if let serde_json::Value::Object(root_object) = special_tokens_map {
                for (_, value) in root_object.iter() {
                    if value.is_string() {
                        tokenizer.add_special_tokens(&[AddedToken {
                            content: value.as_str().unwrap().into(),
                            special: true,
                            ..Default::default()
                        }]);
                    } else if value.is_object() {
                        tokenizer.add_special_tokens(&[AddedToken {
                            content: value["content"].as_str().unwrap().into(),
                            special: true,
                            single_word: value["single_word"].as_bool().unwrap(),
                            lstrip: value["lstrip"].as_bool().unwrap(),
                            rstrip: value["rstrip"].as_bool().unwrap(),
                            normalized: value["normalized"].as_bool().unwrap(),
                        }]);
                    }
                }
            }
            let tokenizer_config: serde_json::Value = match serde_json::from_slice(include_bytes!("../bge_small_en_v15/tokenizer_config.json")) {
                Err(err) => error!("{ERR_PREFIX} Error loading tokenizer config: {err}"),
                Ok(config) => config,
            };
            let model_max_length = match tokenizer_config["model_max_length"].as_f64() {
                None => error!("{ERR_PREFIX} Invalid max model length in tokenizer config"),
                Some(len) => len,
            };

            (tokenizer, model_max_length as i32)
        });

        if !(max_tokens > 0 && max_tokens <= *model_max_length && max_overlap >= 0 && max_overlap < *model_max_length) {
            error!("{ERR_PREFIX} max_tokens must be between 1 and {model_max_length}, and max_overlap must be between 0 and {}", model_max_length - 1);
        }

        let size_config = match ChunkConfig::new(max_tokens as usize).with_overlap(max_overlap as usize) {
            Err(err) => error!("{ERR_PREFIX} Error creating chunk config: {err}"),
            Ok(config) => config,
        };
        let splitter = TextSplitter::new(size_config.with_sizer(tokenizer));
        let chunks = splitter.chunks(document).collect();
        chunks
    })
}

// === Tests ===

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

    #[pg_test]
    fn test_rerank_jina_v1_tiny_en() {
        let candidate_pets = vec!["crocodile", "hamster", "indeterminate", "floorboard", "cat"];
        let scores = crate::rerank_scores_jina_v1_tiny_en("pet", candidate_pets.clone());
        let mut sorted_pets = candidate_pets.clone();
        sorted_pets.sort_by(|pet1, pet2| {
            let index1 = candidate_pets.iter().position(|pet| pet == pet1).unwrap();
            let index2 = candidate_pets.iter().position(|pet| pet == pet2).unwrap();
            scores[index1].partial_cmp(&scores[index2]).unwrap().reverse()
        });
        log!("{:?}", sorted_pets);
        assert!(sorted_pets == vec!["cat", "hamster", "crocodile", "floorboard", "indeterminate"]);
    }

    #[pg_test]
    fn test_chunk_by_characters() {
        assert!(
            crate::chunks_by_characters(
                "The quick brown fox jumps over the lazy dog. In other news, the dish ran away with the spoon.",
                30,
                10
            ) == vec![
                "The quick brown fox jumps over",
                "jumps over the lazy dog.",
                "In other news, the dish ran",
                "dish ran away with the spoon."
            ]
        );
    }

    #[pg_test]
    fn test_chunk_by_tokens() {
        assert!(
            crate::chunks_by_tokens(
                "The quick brown fox jumps over the lazy dog. In other news, the dish ran away with the spoon.",
                8,
                2
            ) == vec![
                "The quick brown fox jumps over the lazy",
                "the lazy dog.",
                "In other news, the dish ran away",
                "ran away with the spoon."
            ]
        );
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
