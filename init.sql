-- Se ejecuta UNA sola vez: solo cuando el volumen de datos está vacío (primer arranque)
CREATE EXTENSION IF NOT EXISTS vector;

-- Tabla de control: evita reprocesar/reinsertar el mismo documento
CREATE TABLE IF NOT EXISTS documents_control (
    id            SERIAL PRIMARY KEY,
    filename      TEXT NOT NULL UNIQUE,
    file_hash     TEXT NOT NULL,
    processed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Chunks vectorizados (nomic-embed-text -> 768 dimensiones)
CREATE TABLE IF NOT EXISTS chunks (
    id                  SERIAL PRIMARY KEY,
    document_filename  TEXT NOT NULL REFERENCES documents_control(filename) ON DELETE CASCADE,
    chunk_index         INTEGER NOT NULL,
    section_title       TEXT,
    content              TEXT NOT NULL,
    embedding             vector(768) NOT NULL,
    article_number       INTEGER  -- NULL si la sección no corresponde a un "Artículo N." explícito
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_article_number_idx
    ON chunks (document_filename, article_number);

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS article_number INTEGER;

CREATE INDEX IF NOT EXISTS chunks_article_number_idx
    ON chunks (document_filename, article_number);
