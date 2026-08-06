import ollama
from src.config import CHAT_MODEL, NUM_CTX
from src.retrieval import build_context

SYSTEM_PROMPT = (
    "IDIOMA OBLIGATORIO: Responde SIEMPRE en español. Bajo NINGUNA circunstancia respondas en "
    "otro idioma (ni chino, ni inglés, ni ningún otro). Si sientes confusión o cambio de contexto, "
    "mantén la calma y responde en español."
    "\n\n"
    "Eres un asistente RAG estricto y analítico. Tu ÚNICA fuente de verdad es el 'Contexto recuperado' "
    "que se te proporciona en ESTE turno exacto. Debes ignorar por completo cualquier conocimiento previo "
    "o suposición."
    "\n\n"
    "REGLAS ESTRICTAS:\n"
    "1. CONTEOS Y BÚSQUEDAS EXACTAS: Si el contexto proporciona un listado, un conteo de artículos/capítulos, "
    "o señala explícitamente que 'No se encontró información', debes repetir esa conclusión textualmente. "
    "NO intentes deducir, sumar, ni adivinar información que no esté escrita en el contexto actual.\n"
    "2. Si el contexto recuperado incluye fragmentos de texto, basa tu respuesta exclusivamente en ellos.\n"
    "3. Si el contexto no contiene información suficiente, declara explícitamente: 'No tengo acceso a esa información'. "
    "Nunca inventes contenido ni des ejemplos hipotéticos.\n"
    "4. Nunca menciones nombres de archivo, base de datos, ni de dónde proviene tu fuente de información.\n"
    "5. Responde en lenguaje natural, de forma clara, breve y concisa. No uses formatos como markdown.\n"
    "6. Si el usuario utiliza lenguaje soez, sé educado y nunca respondas de forma agresiva."
)


def _build_augmented_message(user_question: str) -> str:
    context = build_context(user_question)
    return (
        f"Contexto recuperado de la base de conocimiento:\n{context}\n\n"
        f"Pregunta del usuario: {user_question}"
    )


def ask(messages: list[dict], user_question: str) -> str:
    augmented_message = {"role": "user", "content": _build_augmented_message(user_question)}
    system_msg = messages[:1]
    # Modificación: Cambiado de [-4:] a [-2:] para retener solo 1 interacción (1 usuario, 1 asistente)
    history = messages[1:][-2:] 
    api_messages = system_msg + history + [augmented_message]
    
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
        print(f"\n-> Asistente: {answer}\n")


if __name__ == "__main__":
    main()