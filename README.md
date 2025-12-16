# Cached Sentence Transformer

PostgreSQL-backed embedding cache for [SentenceTransformers](https://www.sbert.net/).

This package provides a small wrapper that caches computed sentence embeddings in a Postgres
table keyed by a stable hash of `(model, normalize flag, text)`, so repeated runs can reuse
stored vectors instead of recomputing them.

## Installation

```bash
pip install cached-sentence-transformer
```

## Quickstart

```python
from cached_sentence_transformer import CachedSentenceTransformer

st = CachedSentenceTransformer(
    model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
    pg_dsn="host=localhost port=5432 dbname=mydb user=myuser password=mypassword",
)

emb = st.encode(["hello", "world"], normalize_embeddings=True)
st.close()
```

## Environment-based DSN

If you do not pass `pg_dsn`, the wrapper will attempt to build it from environment variables
(auto loads environment variables in the .env file in the current working directory) and
will fail fast if any are missing:

- `PSQL_HOST_NAME`
- `PSQL_PORT`
- `PSQL_DBNAME`
- `PSQL_USER`
- `PSQL_PASSWORD`
