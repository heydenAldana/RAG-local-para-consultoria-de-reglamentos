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
                  article_number: int | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunks (document_filename, chunk_index, section_title, content, embedding, article_number)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (filename, chunk_index, section_title, content, embedding, article_number),
        )


def delete_chunks_of_document(conn: psycopg.Connection, filename: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE document_filename = %s", (filename,))


def search_similar_chunks(conn: psycopg.Connection, query_embedding: list[float],
                           top_k: int = 5, document_filter: str | None = None):
    with conn.cursor() as cur:
        where_clause = "WHERE document_filename ILIKE %s" if document_filter else ""
        params = [query_embedding]
        if document_filter:
            params.append(document_filter)
        params += [query_embedding, top_k]
        cur.execute(
            f"""
            SELECT document_filename, section_title, content,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
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
