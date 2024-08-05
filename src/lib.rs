use fastembed::{
    RerankResult, TextEmbedding, TextRerank, TokenizerFiles, UserDefinedEmbeddingModel, UserDefinedRerankingModel,
};
use pgrx::prelude::*;
use std::cell::OnceCell;
use text_splitter::{ChunkConfig, TextSplitter};
use tokenizers::{AddedToken, Tokenizer};

use lopdf::{Bookmark, Document, Object, ObjectId};
use std::collections::BTreeMap;

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
fn _embedding_bge_small_en_v15(input: &str) -> Vec<f32> {
    let vectors = embeddings_bge_small_en_v15(vec![input]);
    match vectors.into_iter().next() {
        None => error!("{ERR_PREFIX} Unexpected empty result vector"),
        Some(vector) => vector,
    }
}

extension_sql!(
    "CREATE FUNCTION embedding_bge_small_en_v15(input text) RETURNS vector(384) 
    LANGUAGE SQL VOLATILE STRICT PARALLEL SAFE AS $$
      SELECT _embedding_bge_small_en_v15(input)::vector(384);
    $$;",
    name = "embedding_bge_small_en_v15_with_cast"
);

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

// === Local splitting/chunking ===

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

// === Local PDF manipulation ===

#[pg_extern(immutable, strict)]
fn pdf_from_pages(files: Vec<&[u8]>) -> Vec<u8> {
    let mut max_id = 1;
    let mut pagenum = 1;

    // Collect all Documents Objects grouped by a map
    let mut documents_pages: BTreeMap<ObjectId, Object> = BTreeMap::new();
    let mut documents_objects: BTreeMap<ObjectId, Object> = BTreeMap::new();
    let mut document = Document::with_version("2.0");

    for file in files {
        let mut doc = match Document::load_mem(file) {
            Err(err) => error!("{ERR_PREFIX} Error opening PDF: {err}"),
            Ok(pdf) => pdf,
        };
        let mut first = false;
        doc.renumber_objects_with(max_id);

        max_id = doc.max_id + 1;

        documents_pages.extend(
            doc.get_pages()
                .into_iter()
                .map(|(_, object_id)| {
                    if !first {
                        let bookmark =
                            Bookmark::new(String::from(format!("Page_{}", pagenum)), [0.0, 0.0, 1.0], 0, object_id);
                        document.add_bookmark(bookmark, None);
                        first = true;
                        pagenum += 1;
                    }

                    (object_id, doc.get_object(object_id).unwrap().to_owned())
                })
                .collect::<BTreeMap<ObjectId, Object>>(),
        );
        documents_objects.extend(doc.objects);
    }

    // Catalog and Pages are mandatory
    let mut catalog_object: Option<(ObjectId, Object)> = None;
    let mut pages_object: Option<(ObjectId, Object)> = None;

    // Process all objects except "Page" type
    for (object_id, object) in documents_objects.iter() {
        // We have to ignore "Page" (as are processed later), "Outlines" and "Outline" objects
        // All other objects should be collected and inserted into the main Document
        match object.type_name().unwrap_or("") {
            "Catalog" => {
                // Collect a first "Catalog" object and use it for the future "Pages"
                catalog_object = Some((
                    if let Some((id, _)) = catalog_object {
                        id
                    } else {
                        *object_id
                    },
                    object.clone(),
                ));
            }
            "Pages" => {
                // Collect and update a first "Pages" object and use it for the future "Catalog"
                // We have also to merge all dictionaries of the old and the new "Pages" object
                if let Ok(dictionary) = object.as_dict() {
                    let mut dictionary = dictionary.clone();
                    if let Some((_, ref object)) = pages_object {
                        if let Ok(old_dictionary) = object.as_dict() {
                            dictionary.extend(old_dictionary);
                        }
                    }
                    pages_object = Some((
                        if let Some((id, _)) = pages_object {
                            id
                        } else {
                            *object_id
                        },
                        Object::Dictionary(dictionary),
                    ));
                }
            }
            "Page" => {}     // Ignored, processed later and separately
            "Outlines" => {} // Ignored, not supported yet
            "Outline" => {}  // Ignored, not supported yet
            _ => {
                document.objects.insert(*object_id, object.clone());
            }
        }
    }

    // If no "Pages" object found abort
    if pages_object.is_none() {
        error!("{ERR_PREFIX} Error loading PDF: no pages root object found");
    }

    // Iterate over all "Page" objects and collect into the parent "Pages" created before
    for (object_id, object) in documents_pages.iter() {
        if let Ok(dictionary) = object.as_dict() {
            let mut dictionary = dictionary.clone();
            dictionary.set("Parent", pages_object.as_ref().unwrap().0);

            document.objects.insert(*object_id, Object::Dictionary(dictionary));
        }
    }

    // If no "Catalog" found abort
    if catalog_object.is_none() {
        error!("{ERR_PREFIX} Error loading PDF: no catalog root object found");
    }

    let catalog_object = catalog_object.unwrap();
    let pages_object = pages_object.unwrap();

    // Build a new "Pages" with updated fields
    if let Ok(dictionary) = pages_object.1.as_dict() {
        let mut dictionary = dictionary.clone();

        // Set new pages count
        dictionary.set("Count", documents_pages.len() as u32);

        // Set new "Kids" list (collected from documents pages) for "Pages"
        dictionary.set(
            "Kids",
            documents_pages
                .into_iter()
                .map(|(object_id, _)| Object::Reference(object_id))
                .collect::<Vec<_>>(),
        );

        document.objects.insert(pages_object.0, Object::Dictionary(dictionary));
    }

    // Build a new "Catalog" with updated fields
    if let Ok(dictionary) = catalog_object.1.as_dict() {
        let mut dictionary = dictionary.clone();
        dictionary.set("Pages", pages_object.0);
        dictionary.remove(b"Outlines"); // Outlines not supported in merged PDFs

        document
            .objects
            .insert(catalog_object.0, Object::Dictionary(dictionary));
    }

    document.trailer.set("Root", catalog_object.0);

    // Update the max internal ID as wasn't updated before due to direct objects insertion
    document.max_id = document.objects.len() as u32;

    // Reorder all new Document objects
    document.renumber_objects();

    // Set any Bookmarks to the First child if they are not set to a page
    document.adjust_zero_pages();

    // Set all bookmarks to the PDF Object tree then set the Outlines to the Bookmark content map.
    if let Some(n) = document.build_outline() {
        if let Ok(x) = document.get_object_mut(catalog_object.0) {
            if let Object::Dictionary(ref mut dict) = x {
                dict.set("Outlines", Object::Reference(n));
            }
        }
    }

    document.compress();

    let mut output = Vec::new();
    match document.save_to(&mut output) {
        Err(err) => error!("{ERR_PREFIX} Error writing PDF: {err}"),
        Ok(x) => x,
    };

    output
}

// === Local HTML to Markdown ===

#[pg_extern(immutable, strict)]
fn markdown_from_html(document: &str) -> String {
    match htmd::convert(document) {
        Err(err) => error!("{ERR_PREFIX} Error converting HTML to Markdown: {err}"),
        Ok(md) => md,
    }
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
        assert!(crate::_embedding_bge_small_en_v15("hello world!").len() == 384);
    }

    #[pg_test]
    fn test_embedding_bge_small_en_v15_immutability() {
        assert!(
            crate::_embedding_bge_small_en_v15("hello world!") == crate::_embedding_bge_small_en_v15("hello world!")
        );
    }

    #[pg_test]
    fn test_embedding_bge_small_en_v15_variability() {
        assert!(crate::_embedding_bge_small_en_v15("hello world!") != crate::_embedding_bge_small_en_v15("bye moon!"));
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
