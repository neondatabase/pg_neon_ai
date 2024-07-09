# Postgres extension to ease use of embeddings

Highly experimental!

Currently supports:

* Querying OpenAI API for embeddings

* Generating embeddings *within Postgres* using https://huggingface.co/Xenova/bge-small-en-v1.5/blob/main/onnx/model.onnx


## Compiling pgvector for PGRX Postgres

```
cd pgvector
export PG_CONFIG=~/.pgrx/13.15/pgrx-install/bin/pg_config
make
make install
```

fastembed-rs cache is at: `~/.pgrx/data-13/.fastembed_cache/`
