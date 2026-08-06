# RAG local para consulta de reglamentos mediante un consultor inteligente

Se explica brevemente en que consiste este repositorio del proyecto de consulta de reglamentos académicos y disciplinarios implementado mediante RAG (Generación Aumentada por Recuperación) local.

*Nota: Este proyecto fue hecho con apoyo de la inteligencia artificial Claude a fin de acelerar el desarrollo de este ya que meramente sirve como demostración para crear un sistema de chat con un consultor inteligente mediante un LLM local. Aún así, sirve muy bien como base. Y de antemano aviso, el archivo `tools.py` no tiene uso real en este proyecto, pero se deja allí y se justifica en la documentación*

---

## Resumen general del proyecto

### ¿En qué consiste?
El proyecto es un pequeño sistema de **Generación Aumentada por Recuperación (RAG)** local diseñado para responder consultas sobre reglamentos institucionales (académicos y disciplinarios). A diferencia de los enfoques RAG convencionales que dependen únicamente de la búsqueda semántica, este sistema implementa una arquitectura de recuperación híbrida que combina:
1. **Búsqueda Híbrida (Vectorial + FTS)**: Lo que hace es que combina la búsqueda por significado (vectorial) con la búsqueda de palabras clave tradicional (Full-Text Search en español) mediante el algoritmo de fusión **Reciprocal Rank Fusion (RRF)** para ordenar los resultados de forma más óptima.
2. **Búsqueda por Coincidencia Exacta**: Búsqueda indexada en base de datos para números de artículos y capítulos específicos, para dar respuestas exactas para consultas puntuales.
3. **Estadísticas Estructurales**: Consultas estructuradas de SQL a través de un registro jerárquico de documentos para resolver de manera determinista preguntas de agregación o conteo de secciones, capítulos, artículos o anexos.

### ¿Cómo funciona?
1. **Conversión e Ingesta**: Los archivos PDF depositados en `rawDocuments/` se analizan mediante su hash SHA256. Si fuesen modificados o son nuevos, se procesan usando la herramienta de Microsoft `MarkItDown` para convertirlos a Markdown estructurado y almacenarlos en `processedDocuments/`.
2. **Segmentación inteligente (Chunking)**: El documento Markdown se divide en secciones utilizando una estrategia en cascada (notese que esto aplica para este contexto. Es posible que usted requiera cambiar la lógica de la segmentación inteligente dependiendo de cómo esté estructurado sus archivos markdown):
   - *Nivel 1*: Encabezados Markdown reales (`#`, `##`, etc.).
   - *Nivel 2*: Expresión regular que detecta `"Artículo N."` en el texto.
   - *Nivel 3 (Fallback)*: Agrupación de párrafos hasta un máximo de 1500 caracteres para evitar chunks sobredimensionados.
3. **Vectorización y Almacenamiento**: Cada fragmento se envía a Ollama para calcular su vector utilizando el modelo `nomic-embed-text` (768 dimensiones). Los vectores, el texto limpio, el título de la sección y el número de artículo (si aplica) se guardan en PostgreSQL.
4. **Recuperación y Chat**: Al iniciar la interfaz en `chat.py`, las preguntas no pasan directamente al LLM. Primero, la lógica en `retrieval.py` intercepta la consulta y genera el contexto idóneo. Dicho contexto enriquecido se inyecta en el prompt enviado al LLM local (Ollama) junto con instrucciones estrictas de no alucinar.

### Tecnologías utilizadas
- Python 3
- PostgreSQL 17 + pgvector
- Ollama
  - **Llama de Chat**: `qwen2.5:7b` (o el configurado en `.env` (use el modelo que desee)).
  - **Lector de Embeddings**: `nomic-embed-text` (este si es necesario, descargalo con el comando `ollama pull nomic-embed-text:latest`).
- Microsoft MarkItDown

### Requisitos

- Python 3 instalado
- Instalar Ollama en tu computadora a fin de poder descargar los modelos locales como `qwen3:4b` (`ollama pull qwen3:4b`) o `quen2.5:7b` (`ollama pull qwen2.5:7b`). Puedes usar el modelo que quieras mientras tu hardware pueda soportarlo al momento de ejecutar sin agobiarse
- Crear un entorno virtual de python de preferencia.
- Docker o Podman instalados

### ¿Cómo se prueba?

1. Crear tu archivo .env (puedes usar el mismo contenido de .env.example)
2. Levante el contenedor:
  -  `docker compose up -d # Docker`
  -  `podman compose up -d # Podman`
3. Levante el servidor de ollama: `ollama serve` 
4.Ejecute `pip install -r requirements.txt` (recomendaría ejecutarlo en un entorno virtual de python con este comando: `python -m venv venv && source venv/bin/activate`) 
5. Ejecute `python -m src.ingest` y espere a que termine de procesar los datos
6. Ejecute `python -m src.chat`.
