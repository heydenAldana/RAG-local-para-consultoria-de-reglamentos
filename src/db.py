import psycopg
from pgvector.psycopg import register_vector
from src.config import DB_DSN


def get_connection() -> psycopg.Connection:
    conn = psycopg.connect(DB_DSN, autocommit=False)
    register_vector(conn)
    return conn


def document_already_processed(conn: psycopg.Connection, filename: str, file_hash: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT file_hash FROM documents_control WHERE filename = %s",
            (filename,),
        )
        row = cur.fetchone()
        return row is not None and row[0] == file_hash


def register_document(conn: psycopg.Connection, filename: str, file_hash: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents_control (filename, file_hash)
            VALUES (%s, %s)
            ON CONFLICT (filename) DO UPDATE SET file_hash = EXCLUDED.file_hash,
                                                  processed_at = now()
            """,
            (filename, file_hash),
        )


def insert_chunk(conn: psycopg.Connection, filename: str, chunk_index: int,
                  section_title: str, content: str, embedding: list[float],
                  article_number: int | None = None,
                  chapter_number: int | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunks (document_filename, chunk_index, section_title, content, embedding,
                                article_number, chapter_number)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (filename, chunk_index, section_title, content, embedding, article_number, chapter_number),
        )


def delete_chunks_of_document(conn: psycopg.Connection, filename: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE document_filename = %s", (filename,))


def search_similar_chunks(conn: psycopg.Connection, query_embedding: list[float],
                           query_text: str, top_k: int = 5,
                           document_filter: str | None = None, rrf_k: int = 60):
    with conn.cursor() as cur:
        doc_filter = "AND document_filename ILIKE %(doc_filter)s" if document_filter else ""
        pool = top_k * 5
        params = {
            "emb": query_embedding,
            "q": query_text,
            "pool": pool,
            "top_k": top_k,
            "rrf_k": rrf_k,
        }
        if document_filter:
            params["doc_filter"] = document_filter
        cur.execute(
            f"""
            WITH vec AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(emb)s::vector) AS rank_v
                FROM chunks
                WHERE TRUE {doc_filter}
                ORDER BY embedding <=> %(emb)s::vector
                LIMIT %(pool)s
            ),
            fts AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(content_fts, q) DESC) AS rank_f
                FROM chunks, websearch_to_tsquery('spanish', %(q)s) q
                WHERE content_fts @@ q {doc_filter}
                ORDER BY ts_rank_cd(content_fts, q) DESC
                LIMIT %(pool)s
            ),
            combined AS (
                SELECT COALESCE(v.id, f.id) AS id,
                       COALESCE(1.0 / (%(rrf_k)s + v.rank_v), 0) +
                       COALESCE(1.0 / (%(rrf_k)s + f.rank_f), 0) AS rrf_score
                FROM vec v FULL OUTER JOIN fts f ON v.id = f.id
            )
            SELECT c.document_filename, c.section_title, c.content, cb.rrf_score
            FROM combined cb
            JOIN chunks c ON c.id = cb.id
            ORDER BY cb.rrf_score DESC
            LIMIT %(top_k)s
            """,
            params,
        )
        return cur.fetchall()


def get_chunks_by_article_number(conn: psycopg.Connection, article_number: int,
                                  document_filter: str | None = None):
    """Búsqueda EXACTA (no semántica) por número de artículo. Evita confundir art. 5 con art. 35."""
    with conn.cursor() as cur:
        if document_filter:
            cur.execute(
                """
                SELECT document_filename, section_title, content
                FROM chunks
                WHERE article_number = %s AND document_filename ILIKE %s
                ORDER BY document_filename, chunk_index
                """,
                (article_number, document_filter),
            )
        else:
            cur.execute(
                """
                SELECT document_filename, section_title, content
                FROM chunks
                WHERE article_number = %s
                ORDER BY document_filename, chunk_index
                """,
                (article_number,),
            )
        return cur.fetchall()


def get_chunks_by_chapter_number(conn: psycopg.Connection, chapter_number: int,
                                  document_filter: str | None = None):
    with conn.cursor() as cur:
        if document_filter:
            cur.execute(
                """
                SELECT document_filename, section_title, content
                FROM chunks
                WHERE chapter_number = %s AND document_filename ILIKE %s
                ORDER BY document_filename, chunk_index
                """,
                (chapter_number, document_filter),
            )
        else:
            cur.execute(
                """
                SELECT document_filename, section_title, content
                FROM chunks
                WHERE chapter_number = %s
                ORDER BY document_filename, chunk_index
                """,
                (chapter_number,),
            )
        return cur.fetchall()


def get_document_stats(conn: psycopg.Connection):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT document_filename, COUNT(DISTINCT article_number), MAX(article_number)
            FROM chunks
            WHERE article_number IS NOT NULL
            GROUP BY document_filename
            ORDER BY document_filename
            """
        )
        return cur.fetchall()


def delete_structure_of_document(conn: psycopg.Connection, filename: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM document_structure WHERE document_filename = %s", (filename,))


def insert_structure_entries(conn: psycopg.Connection, filename: str,
                              entries: list[tuple[str, int, str]]) -> None:
    if not entries:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO document_structure (document_filename, entity_type, entity_number, entity_label)
            VALUES (%s, %s, %s, %s)
            """,
            [(filename, entity_type, number, label) for entity_type, number, label in entries],
        )


def get_structure_stats(conn: psycopg.Connection, entity_type: str,
                         document_filter: str | None = None):
    with conn.cursor() as cur:
        if document_filter:
            cur.execute(
                """
                SELECT document_filename, COUNT(DISTINCT entity_number), MAX(entity_number)
                FROM document_structure
                WHERE entity_type = %s AND document_filename ILIKE %s
                GROUP BY document_filename
                ORDER BY document_filename
                """,
                (entity_type, document_filter),
            )
        else:
            cur.execute(
                """
                SELECT document_filename, COUNT(DISTINCT entity_number), MAX(entity_number)
                FROM document_structure
                WHERE entity_type = %s
                GROUP BY document_filename
                ORDER BY document_filename
                """,
                (entity_type,),
            )
        return cur.fetchall()
