import sys

import ollama
from src.config import EMBED_MODEL
from src.db import get_connection, search_similar_chunks

# Definición del esquema de la tool en formato compatible con Ollama tool-calling
SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Busca información relevante en la base de conocimiento vectorizada "
            "(documentos PDF procesados). Úsala siempre que el usuario pregunte "
            "algo que pueda estar contenido en esos documentos."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Consulta en lenguaje natural para buscar en los documentos.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Número de fragmentos relevantes a devolver (default 5).",
                },
            },
            "required": ["query"],
        },
    },
}


def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Ejecuta la búsqueda vectorial real y formatea el resultado como texto para el LLM."""
    embed_response = ollama.embed(model=EMBED_MODEL, input=query)
    query_embedding = embed_response["embeddings"][0]
    conn = get_connection()
    try:
        results = search_similar_chunks(conn, query_embedding, top_k=top_k)
    finally:
        conn.close()
    # DEBUG temporal: muestra qué recuperó realmente la búsqueda vectorial
    print(f"\n[DEBUG] Query: '{query}' -> {len(results)} chunks recuperados:", file=sys.stderr)
    for filename, section_title, content, similarity in results:
        preview = content[:80].replace("\n", " ")
        print(f"  - sim={similarity:.3f} | {filename} | sección: '{section_title}' | {preview}...", file=sys.stderr)
    if not results:
        return "No se encontró información relevante en la base de conocimiento."
    formatted = []
    for filename, section_title, content, similarity in results:
        formatted.append(
            f"[Fuente: {filename} | Sección: {section_title} | similitud: {similarity:.2f}]\n{content}"
        )
    return "\n\n---\n\n".join(formatted)

# Mapa nombre_tool -> función Python real (usado por el loop de chat)
AVAILABLE_TOOLS = {
    "search_knowledge_base": search_knowledge_base,
}
