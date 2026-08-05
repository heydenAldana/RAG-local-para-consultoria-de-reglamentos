import re
import sys
import unicodedata

import ollama
from src.config import EMBED_MODEL
from src.db import (
    get_connection,
    search_similar_chunks,
    get_chunks_by_article_number,
    get_structure_stats,
)

ARTICLE_QUERY_PATTERN = re.compile(r"art[ií]culo\s+(\d+)", re.IGNORECASE)

COUNT_QUERY_PATTERN = re.compile(r"cu[aá]nt[oa]s?\s+([a-záéíóúñ]+)", re.IGNORECASE)

ENTITY_SYNONYMS = {
    "articulo": "articulo", "articulos": "articulo",
    "capitulo": "capitulo", "capitulos": "capitulo",
    "seccion": "seccion", "secciones": "seccion",
    "anexo": "anexo", "anexos": "anexo",
}

DOCUMENT_KEYWORDS = {
    "disciplinario": "%disciplinario%",
    "disciplina": "%disciplinario%",
    "academico": "%academico%",
    "académico": "%academico%",
}


def _normalize_word(word: str) -> str:
    return unicodedata.normalize("NFKD", word).encode("ascii", "ignore").decode().lower()


def _detect_document_filter(query: str) -> str | None:
    q = query.lower()
    for keyword, sql_pattern in DOCUMENT_KEYWORDS.items():
        if keyword in q:
            return sql_pattern
    return None


def _debug(label: str, details: str) -> None:
    print(f"\n[DEBUG] {label}: {details}", file=sys.stderr)


def _semantic_search(query: str, document_filter: str | None, top_k: int = 15) -> str:
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

# Esta funcón ayuda a determnar si se trata de un conteo simple o un conteo temático
def _is_simple_count(query: str, match: re.Match) -> bool:
    remainder = query[:match.start()] + query[match.end():]
    remainder = re.sub(r"[¿?.,;:!¡\s]+", " ", remainder).strip()
    stopwords = {
        "hay", "tiene", "tienen", "son", "hay", "existen", "existe",
        "el", "la", "los", "las", "un", "una", "unos", "unas",
        "del", "de", "en", "total", "reglamento", "documento",
    }
    meaningful = [w for w in remainder.lower().split() if _normalize_word(w) not in stopwords]
    return len(meaningful) == 0


def build_context(user_question: str) -> str:
    document_filter = _detect_document_filter(user_question)
    # 1) Preguntas de conteo/agregación sobre una entidad estructural conocida
    count_match = COUNT_QUERY_PATTERN.search(user_question)
    if count_match:
        noun = _normalize_word(count_match.group(1))
        entity_type = ENTITY_SYNONYMS.get(noun)
        if entity_type:
            if _is_simple_count(user_question, count_match):
                conn = get_connection()
                try:
                    stats = get_structure_stats(conn, entity_type, document_filter)
                finally:
                    conn.close()
                _debug("Conteo estructural", f"tipo='{entity_type}' -> {stats}")
                if not stats:
                    return f"No se encontró información sobre la cantidad de {noun} en la base de conocimiento."
                lines = [
                    f"- {filename}: {count} {entity_type}s identificados (numeración máxima detectada: {max_num})"
                    for filename, count, max_num in stats
                ]
                return f"Conteo de {entity_type}s por documento:\n" + "\n".join(lines)
            _debug("Conteo temático", f"tipo='{entity_type}', derivando a búsqueda semántica")
            return _semantic_search(user_question, document_filter)
        _debug("Conteo no soportado", f"'{noun}' no está en ENTITY_SYNONYMS")
        semantic = _semantic_search(user_question, document_filter)
        supported = ", ".join(sorted(set(ENTITY_SYNONYMS.values())))
        return (
            f"No es posible dar un conteo exacto de '{noun}': solo se pueden contar con precisión "
            f"estas divisiones estructurales del documento: {supported}. "
            f"Aquí hay fragmentos relacionados por búsqueda semántica, pero NO representan un conteo real:\n\n"
            f"{semantic}"
        )
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
        # Si no hubo match exacto, se avisa explícitamente para que el modelo no finja que sí lo encontró
        semantic = _semantic_search(user_question, document_filter)
        return (
            f"No se encontró exactamente el artículo {article_number} en la base de conocimiento. "
            f"Esto es lo más cercano por búsqueda semántica (puede NO corresponder al artículo pedido):\n\n{semantic}"
        )

    # 3) Pregunta abierta (búsqueda semántica normal)
    return _semantic_search(user_question, document_filter)
