import re
import sys

import ollama
from src.config import EMBED_MODEL
from src.db import (
    get_connection,
    search_similar_chunks,
    get_chunks_by_article_number,
    get_document_stats,
)

ARTICLE_QUERY_PATTERN = re.compile(r"art[ií]culo\s+(\d+)", re.IGNORECASE)
COUNT_QUERY_PATTERN = re.compile(r"cu[aá]nt[oa]s?\s+art[ií]culos", re.IGNORECASE)

"""
Palabras clave simples para acotar la búsqueda a un documento si el usuario lo menciona
Si la información fuese de un distinto contexto, se deben cambiar las palabras también.
"""
DOCUMENT_KEYWORDS = {
    "disciplinario": "%disciplinario%",
    "disciplina": "%disciplinario%",
    "academico": "%academico%",
    "académico": "%academico%",
}


def _detect_document_filter(query: str) -> str | None:
    q = query.lower()
    for keyword, sql_pattern in DOCUMENT_KEYWORDS.items():
        if keyword in q:
            return sql_pattern
    return None


def _debug(label: str, details: str) -> None:
    print(f"\n[DEBUG] {label}: {details}", file=sys.stderr)


def _semantic_search(query: str, document_filter: str | None, top_k: int = 5) -> str:
    embed_response = ollama.embed(model=EMBED_MODEL, input=query)
    query_embedding = embed_response["embeddings"][0]
    conn = get_connection()
    try:
        results = search_similar_chunks(conn, query_embedding, top_k=top_k, document_filter=document_filter)
    finally:
        conn.close()
    _debug("Búsqueda semántica", f"'{query}' -> {len(results)} chunks")
    for filename, section_title, content, similarity in results:
        preview = content[:80].replace("\n", " ")
        print(f"  - sim={similarity:.3f} | {filename} | sección: '{section_title}' | {preview}...", file=sys.stderr)
    if not results:
        return "No se encontró información relevante en la base de conocimiento."
    return "\n\n---\n\n".join(
        f"[Sección: {section_title} | similitud: {similarity:.2f}]\n{content}"
        for _, section_title, content, similarity in results
    )


def build_context(user_question: str) -> str:
    document_filter = _detect_document_filter(user_question)
    # 1) Preguntas de conteo/agregación
    if COUNT_QUERY_PATTERN.search(user_question):
        conn = get_connection()
        try:
            stats = get_document_stats(conn)
        finally:
            conn.close()
        _debug("Conteo de artículos", str(stats))
        if not stats:
            return "No se encontró información sobre la cantidad de artículos en la base de conocimiento."
        lines = [
            f"- {filename}: {count} artículos identificados (numeración máxima detectada: {max_num})"
            for filename, count, max_num in stats
        ]
        return "Conteo de artículos por documento:\n" + "\n".join(lines)
    # 2) Número de artículo explícito (match EXACTO)
    article_match = ARTICLE_QUERY_PATTERN.search(user_question)
    if article_match:
        article_number = int(article_match.group(1))
        conn = get_connection()
        try:
            exact_results = get_chunks_by_article_number(conn, article_number, document_filter)
        finally:
            conn.close()
        _debug("Búsqueda exacta por artículo", f"artículo {article_number} -> {len(exact_results)} coincidencias")
        if exact_results:
            return "\n\n---\n\n".join(
                f"[Documento: {filename} | Sección: {section_title}]\n{content}"
                for filename, section_title, content in exact_results
            )
        # No hubo match exacto: se avisa explícitamente para que el modelo no finja que sí lo encontró
        semantic = _semantic_search(user_question, document_filter)
        return (
            f"No encontré información exacta sobre el artículo {article_number} en mis fuentes de información. "
            f"Esto es lo más cercano que encontré relacionado con (puede NO corresponder al artículo pedido):\n\n{semantic}"
        )
    # 3) Pregunta abierta -> búsqueda semántica normal
    return _semantic_search(user_question, document_filter)
