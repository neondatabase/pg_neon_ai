# Postgres extension to ease use of embeddings

## Compiling pgvector for PGRX Postgres

cd pgvector
export PG_CONFIG=~/.pgrx/13.15/pgrx-install/bin/pg_config
make
make install
