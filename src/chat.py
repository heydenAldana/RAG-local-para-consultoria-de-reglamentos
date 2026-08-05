import ollama
from src.config import CHAT_MODEL, NUM_CTX
from src.retrieval import build_context

SYSTEM_PROMPT = (
    "Eres un asistente que responde preguntas basándote ÚNICAMENTE en el contexto que se te "
    "proporciona en cada turno, extraído de una base de conocimiento local de documentos "
    "relacionados al reglamento disciplinario y académico. "
    "Si el contexto no contiene información suficiente para responder, dile al usuario que no "
    "tienes información disponible para su consulta y discúlpate; nunca inventes contenido ni "
    "des ejemplos hipotéticos. "
    "Responde siempre en español, de forma clara, breve y concisa. Nunca respondas en otro idioma. "
    "Si el usuario utiliza lenguaje soez contra ti o en general, sé educado y nunca le respondas "
    "de forma agresiva o con insultos. Nunca menciones nombres de archivo ni de dónde proviene tu "
    "fuente de información. Responde siempre en lenguaje natural, y nunca uses otros formatos."
)


def _build_augmented_message(user_question: str) -> str:
    context = build_context(user_question)
    return (
        f"Contexto recuperado de la base de conocimiento:\n{context}\n\n"
        f"Pregunta del usuario: {user_question}"
    )


def ask(messages: list[dict], user_question: str) -> str:
    augmented_message = {"role": "user", "content": _build_augmented_message(user_question)}
    api_messages = messages + [augmented_message]
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=api_messages,
        options={"num_ctx": NUM_CTX},
        think=False,
    )
    answer = response["message"]["content"]
    messages.append({"role": "user", "content": user_question})
    messages.append({"role": "assistant", "content": answer})
    return answer


def main():
    print(f"Chat RAG local (modelo: {CHAT_MODEL}). Escribe 'salir' para terminar.\n")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        user_input = input("Tú: ").strip()
        if user_input.lower() in {"salir", "exit", "quit"}:
            print("Hasta luego.")
            break
        if not user_input:
            continue
        answer = ask(messages, user_input)
        print(f"\nAsistente: {answer}\n")


if __name__ == "__main__":
    main()
