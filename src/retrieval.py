import re
import sys
import unicodedata

import ollama
from src.config import EMBED_MODEL
from src.db import (
    get_connection,
    search_similar_chunks,
    get_chunks_by_article_number,
    get_chunks_by_chapter_number,
    get_structure_stats,
)

ARTICLE_QUERY_PATTERN = re.compile(r"art[ií]culo\s+(\d+)", re.IGNORECASE)

CHAPTER_QUERY_PATTERN = re.compile(r"cap[ií]tulo\s+(\d+)", re.IGNORECASE)

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
        results = search_similar_chunks(conn, query_embedding, query_text=query,
                                        top_k=top_k, document_filter=document_filter)
    finally:
        conn.close()
    _debug("Búsqueda híbrida (RRF)", f"'{query}' -> {len(results)} chunks")
    for row in results:
        filename, section_title, content, rrf_score = row[:4]
        preview = content[:80].replace("\n", " ")
        print(f"  - rrf={rrf_score:.4f} | {filename} | sección: '{section_title}' | {preview}...", file=sys.stderr)
    if not results:
        return "No se encontró información relevante en la base de conocimiento."
    return "\n\n---\n\n".join(
        f"[Sección: {row[1]} | relevancia: {row[3]:.4f}]\n{row[2]}"
        for row in results
    )


def _is_simple_count(query: str, match: re.Match) -> bool:
    remainder = query[:match.start()] + query[match.end():]
    remainder = re.sub(r"[¿?.,;:!¡\s]+", " ", remainder).strip()
    stopwords = {
        "hay", "tiene", "tienen", "son", "existen", "existe",
        "el", "la", "los", "las", "un", "una", "unos", "unas",
        "del", "de", "en", "total", "reglamento", "documento",
        "hablan", "habla", "sobre", "acerca", "tratan", "trata",
        "mencionan", "menciona", "dicen", "dice", "cuales", "consisten", "que", "y", "o"
    }
    for k in DOCUMENT_KEYWORDS:
        stopwords.add(_normalize_word(k))
    meaningful = [w for w in remainder.lower().split() if _normalize_word(w) not in stopwords]
    return len(meaningful) == 0

def _thematic_count_and_search(query: str, noun: str, entity_type: str, document_filter: str | None) -> str:
    embed_response = ollama.embed(model=EMBED_MODEL, input=query)
    query_embedding = embed_response["embeddings"][0]
    conn = get_connection()
    try:
        results = search_similar_chunks(conn, query_embedding, query_text=query,
                                        top_k=40, document_filter=document_filter)
    finally:
        conn.close()
    _debug("Búsqueda temática y conteo", f"'{query}' -> {len(results)} chunks explorados")
    if not results:
        return f"No se encontró información para contar {noun} sobre este tema."
    entities_found = set()
    for row in results:
        filename, section_title, content, rrf_score, art_num, chap_num = row
        if entity_type == "articulo" and art_num:
            entities_found.add((filename, f"Artículo {art_num}"))
        elif entity_type == "capitulo" and chap_num:
            entities_found.add((filename, f"Capítulo {chap_num}"))
    summary = ""
    if entities_found:
        summary += f"Se han identificado {len(entities_found)} {entity_type}s relevantes a la consulta:\n"
        for doc_name, entity_name in sorted(entities_found, key=lambda x: (x[0], int(re.search(r'\d+', x[1]).group()) if re.search(r'\d+', x[1]) else 0)):
            summary += f"- {entity_name} (en {doc_name})\n"
        summary += "\nA continuación se presenta el contenido de los fragmentos más relevantes:\n\n"
    else:
        summary += f"No se pudieron identificar números exactos de {entity_type}s, pero aquí están los fragmentos más relevantes:\n\n"
    chunks_text = "\n\n---\n\n".join(
        f"[Sección: {row[1]} | relevancia: {row[3]:.4f}]\n{row[2]}"
        for row in results[:15]
    )
    return summary + chunks_text


def build_context(user_question: str) -> str:
    document_filter = _detect_document_filter(user_question)
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
            _debug("Conteo temático", f"tipo='{entity_type}', derivando a búsqueda temática")
            return _thematic_count_and_search(user_question, noun, entity_type, document_filter)
        _debug("Conteo no soportado", f"'{noun}' no está en ENTITY_SYNONYMS")
        semantic = _semantic_search(user_question, document_filter)
        supported = ", ".join(sorted(set(ENTITY_SYNONYMS.values())))
        return (
            f"No es posible dar un conteo exacto de '{noun}': solo se pueden contar con precisión "
            f"estas divisiones estructurales del documento: {supported}. "
            f"Aquí hay fragmentos relacionados por búsqueda semántica, pero NO representan un conteo real:\n\n"
            f"{semantic}"
        )
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
        semantic = _semantic_search(user_question, document_filter)
        return (
            f"No se encontró exactamente el artículo {article_number} en la base de conocimiento. "
            f"Esto es lo más cercano por búsqueda semántica (puede NO corresponder al artículo pedido):\n\n{semantic}"
        )

    chapter_match = CHAPTER_QUERY_PATTERN.search(user_question)
    if chapter_match:
        chapter_number = int(chapter_match.group(1))
        conn = get_connection()
        try:
            exact_results = get_chunks_by_chapter_number(conn, chapter_number, document_filter)
        finally:
            conn.close()
        _debug("Búsqueda exacta por capítulo", f"capítulo {chapter_number} -> {len(exact_results)} coincidencias")
        if exact_results:
            return "\n\n---\n\n".join(
                f"[Documento: {filename} | Sección: {section_title}]\n{content}"
                for filename, section_title, content in exact_results
            )
        semantic = _semantic_search(user_question, document_filter)
        return (
            f"No se encontró exactamente el capítulo {chapter_number} en la base de conocimiento. "
            f"Esto es lo más cercano por búsqueda semántica (puede NO corresponder al capítulo pedido):\n\n{semantic}"
        )

    return _semantic_search(user_question, document_filter)
