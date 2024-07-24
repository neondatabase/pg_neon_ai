# Postgres extension to ease use of embeddings

Highly experimental!

Currently supports:

* Querying OpenAI API for embeddings

* Chunking *within Postgres* for https://huggingface.co/Xenova/bge-small-en-v1.5

* Generating embeddings *within Postgres* using https://huggingface.co/Xenova/bge-small-en-v1.5

* Reranking *within Postgres* using https://huggingface.co/jinaai/jina-reranker-v1-tiny-en

## Compiling pgvector for PGRX Postgres

```
cd pgvector
export PG_CONFIG=~/.pgrx/13.15/pgrx-install/bin/pg_config
make
make install
```

fastembed-rs cache is at: `~/.pgrx/data-13/.fastembed_cache/`
