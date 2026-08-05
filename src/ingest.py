import hashlib
import re
import sys

import ollama
from markitdown import MarkItDown

from src.config import RAW_DOCS_DIR, PROCESSED_DOCS_DIR, EMBED_MODEL
from src.db import (
    get_connection,
    document_already_processed,
    register_document,
    insert_chunk,
    delete_chunks_of_document,
)

HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

ARTICLE_PATTERN = re.compile(r"^\s*(Art[ií]culo\s+\d+[o°]?\.?)", re.MULTILINE | re.IGNORECASE)

MAX_CHUNK_CHARS = 1500


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def convert_to_markdown(pdf_path) -> str:
    md_converter = MarkItDown()
    result = md_converter.convert(str(pdf_path))
    md_path = PROCESSED_DOCS_DIR / (pdf_path.stem + ".md")
    md_path.write_text(result.text_content, encoding="utf-8")
    return result.text_content


def _split_by_pattern(text: str, pattern: re.Pattern, title_from_match: bool) -> list[tuple[str, str]]:
    """Corta 'text' en los puntos donde matchea 'pattern'. Cada match arranca una sección nueva."""
    matches = list(pattern.finditer(text))
    sections = []
    for i, match in enumerate(matches):
        title = match.group(1).strip() if title_from_match else match.group(2).strip()
        start = match.start() if title_from_match else match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections.append((title, content))
    return sections


def _split_by_paragraphs(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[tuple[str, str]]:
    """Último recurso: agrupa párrafos consecutivos hasta un límite de tamaño por chunk."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sections = []
    buffer = ""
    idx = 1
    for p in paragraphs:
        candidate = f"{buffer}\n\n{p}" if buffer else p
        if len(candidate) > max_chars and buffer:
            sections.append((f"Fragmento {idx}", buffer.strip()))
            idx += 1
            buffer = p
        else:
            buffer = candidate
    if buffer.strip():
        sections.append((f"Fragmento {idx}", buffer.strip()))
    return sections


def _extract_article_number(title: str) -> int | None:
    match = re.search(r"(\d+)", title)
    return int(match.group(1)) if match else None


def split_by_sections(markdown_text: str) -> list[tuple[str, str, int | None]]:
    """
    Devuelve lista de (titulo_seccion, contenido, numero_articulo), probando en cascada:
    1) Encabezados Markdown reales (#, ##, ...)
    2) Patrón 'Artículo N.' (común en reglamentos convertidos a texto plano) -> guarda el número
    3) Párrafos agrupados por tamaño máximo (fallback genérico)
    Evita que un documento sin encabezados termine como un único chunk gigante.
    """
    if not markdown_text.strip():
        return []
    header_matches = list(HEADER_PATTERN.finditer(markdown_text))
    if len(header_matches) >= 2:
        sections = _split_by_pattern(markdown_text, HEADER_PATTERN, title_from_match=False)
        return [(title, content, None) for title, content in sections]
    article_matches = list(ARTICLE_PATTERN.finditer(markdown_text))
    if len(article_matches) >= 2:
        sections = _split_by_pattern(markdown_text, ARTICLE_PATTERN, title_from_match=True)
        return [(title, content, _extract_article_number(title)) for title, content in sections]
    sections = _split_by_paragraphs(markdown_text)
    return [(title, content, None) for title, content in sections]


def embed_text(text: str) -> list[float]:
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]


def ingest_document(pdf_path, force: bool = False):
    filename = pdf_path.name
    h = file_hash(pdf_path)
    conn = get_connection()
    try:
        if not force and document_already_processed(conn, filename, h):
            print(f"[SKIP] '{filename}' ya estaba procesado (sin cambios). Usa --force para re-procesar.")
            return
        print(f"[PROCESANDO] {filename}")
        markdown_text = convert_to_markdown(pdf_path)
        sections = split_by_sections(markdown_text)
        # Si el documento existía con otro hash (cambió), se limpian sus chunks previos
        delete_chunks_of_document(conn, filename)
        register_document(conn, filename, h)
        for idx, (title, content, article_number) in enumerate(sections):
            embedding = embed_text(content)
            insert_chunk(conn, filename, idx, title, content, embedding, article_number=article_number)
        conn.commit()
        print(f"[OK] {filename}: {len(sections)} secciones vectorizadas e insertadas.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    force = "--force" in sys.argv
    pdf_files = sorted(RAW_DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No se encontraron PDFs en {RAW_DOCS_DIR}")
        sys.exit(0)
    for pdf_path in pdf_files:
        ingest_document(pdf_path, force=force)


if __name__ == "__main__":
    main()
