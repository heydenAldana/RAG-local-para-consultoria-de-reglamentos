# Documentación del Proyecto: RAG Local para Consulta de Reglamentos

**Nota**: La documentación fue escrita y en gran parte generada con ayuda de la IA agéntica Gemini Flash v3.5 (Antigravity)

Este documento proporciona una visión general y técnica del proyecto de consulta de reglamentos académicos y disciplinarios implementado mediante RAG (Generación Aumentada por Recuperación) local.

---

## 1. Resumen General del Proyecto

### ¿En qué consiste?
El proyecto es un sistema de **Generación Aumentada por Recuperación (RAG)** local diseñado para responder consultas sobre reglamentos institucionales (académicos y disciplinarios). A diferencia de los enfoques RAG convencionales que dependen únicamente de la búsqueda semántica —la cual suele fallar o confundirse al buscar números de artículos específicos o realizar conteos—, este sistema implementa una arquitectura de recuperación híbrida que combina:
1. **Búsqueda Semántica**: Búsqueda vectorial mediante embeddings para preguntas abiertas.
2. **Búsqueda por Coincidencia Exacta**: Búsqueda indexada en base de datos para números de artículos específicos.
3. **Estadísticas Agregadas**: Consultas estructuradas de SQL para preguntas de agregación o conteo.

### ¿Cómo funciona?
El ciclo de vida del dato consta de las siguientes fases:
1. **Conversión e Ingesta**: Los archivos PDF depositados en `rawDocuments/` se analizan mediante su hash SHA256. Si sufrieron modificaciones o son nuevos, se procesan usando la herramienta de Microsoft `MarkItDown` para convertirlos a Markdown estructurado y almacenarlos en `processedDocuments/`.
2. **Segmentación inteligente (Chunking)**: El documento Markdown se divide en secciones utilizando una estrategia en cascada:
   - *Nivel 1*: Encabezados Markdown reales (`#`, `##`, etc.).
   - *Nivel 2*: Expresión regular que detecta `"Artículo N."` en el texto.
   - *Nivel 3 (Fallback)*: Agrupación de párrafos hasta un máximo de 1500 caracteres para evitar chunks sobredimensionados.
3. **Vectorización y Almacenamiento**: Cada fragmento se envía a Ollama para calcular su vector utilizando el modelo `nomic-embed-text` (768 dimensiones). Los vectores, el texto limpio, el título de la sección y el número de artículo (si aplica) se guardan en PostgreSQL.
4. **Recuperación y Chat**: Al iniciar la interfaz en `chat.py`, las preguntas no pasan directamente al LLM. Primero, la lógica en `retrieval.py` intercepta la consulta y genera el contexto idóneo. Dicho contexto enriquecido se inyecta en el prompt enviado al LLM local (Ollama) junto con instrucciones estrictas de no alucinar.

### Tecnologías Utilizadas
- **Python 3**: Lenguaje base para los scripts de ingesta y conversación.
- **PostgreSQL 17 + pgvector**: Base de datos relacional y vectorial. Utiliza el índice `HNSW` (Hierarchical Navigable Small World) para acelerar búsquedas vectoriales y un índice estándar B-Tree para el número de artículo exacto.
- **Docker & Docker Compose**: Automatiza el despliegue del motor PostgreSQL en un entorno aislado y reproducible.
- **Ollama**: Servidor local para ejecutar y consumir modelos de IA:
  - **Llama de Chat**: `qwen2.5:7b` (o el configurado en `.env`).
  - **Lector de Embeddings**: `nomic-embed-text`.
- **Microsoft MarkItDown**: Extractor de texto que simplifica la conversión de PDFs a Markdown estructurado.

### ¿Cómo se prueba?
El proyecto se valida de manera empírica a través de ejecuciones de sus scripts clave:
1. **Prueba de Ingesta**: Ejecutando `python -m src.ingest`. Se verifica que detecte nuevos archivos PDF, cree su equivalente Markdown y cargue correctamente la base de datos.
2. **Prueba de Conversación (Chat)**: Ejecutando `python -m src.chat`. Para comprobar la correcta lógica de recuperación, se pueden evaluar tres comportamientos:
   - **Preguntas de conteo**: *"¿Cuántos artículos tiene el reglamento disciplinario?"* (deberá activar la función de conteo de base de datos).
   - **Preguntas de artículo exacto**: *"¿Qué dice el artículo 15?"* o *"Artículo 5 del reglamento académico"* (deberá retornar exactamente la sección indicada, sin variaciones semánticas erróneas).
   - **Preguntas libres**: Consultas generales sobre sanciones o deberes para comprobar la búsqueda semántica HNSW.

---

## 2. Funcionamiento y Estructura del Código (`./src`)

La lógica central y los módulos del proyecto se encuentran agrupados bajo el directorio `./src`. A continuación, se detalla la responsabilidad de cada archivo:

### `__init__.py`
Permite que Python reconozca el directorio `src` como un paquete del sistema, facilitando las importaciones relativas y absolutas entre módulos.

### `config.py`
Módulo de configuración centralizada. Se encarga de:
- Cargar variables del entorno desde el archivo `.env`.
- Resolver las rutas del proyecto a nivel de sistema operativo (`rawDocuments/` y `processedDocuments/`).
- Centralizar las credenciales de base de datos (DSN) y configuraciones de Ollama (modelos a usar y tamaño de la ventana de contexto).

### `db.py`
Capa de persistencia y comunicación con PostgreSQL. Implementa consultas con el driver nativo de Python `psycopg` (versión 3) y la integración de `pgvector`. Sus funciones clave son:
- `get_connection()`: Abre conexiones y registra el adaptador vectorial para psycopg.
- `document_already_processed()` y `register_document()`: Previenen la duplicidad y reprocesamiento innecesario de archivos usando hashes.
- `insert_chunk()` y `delete_chunks_of_document()`: Gestión del ciclo de vida de los fragmentos en la tabla `chunks`.
- `search_similar_chunks()`: Ejecuta búsquedas de vecinos más cercanos utilizando distancia de coseno en `pgvector`.
- `get_chunks_by_article_number()`: Recupera de forma determinista un artículo específico basándose en su índice de base de datos.
- `get_document_stats()`: Cuenta la cantidad de artículos únicos detectados por documento para responder preguntas analíticas.

### `ingest.py`
Pipeline de procesamiento de documentos. Es el punto de entrada para preparar la base de datos:
1. Escanea la carpeta `rawDocuments/`.
2. Valida modificaciones mediante SHA256.
3. Convierte archivos PDF a Markdown con `MarkItDown`.
4. Divide el contenido en fragmentos utilizando la estrategia en cascada (`split_by_sections`).
5. Genera embeddings llamando a Ollama y almacena todo en PostgreSQL.

### `retrieval.py`
Constituye la inteligencia del motor RAG. Implementa `build_context`, una función determinista que analiza sintácticamente la pregunta del usuario mediante expresiones regulares antes de consultar el LLM:
- **Detección de conteo**: Si la consulta pregunta por cantidades de artículos, llama a `get_document_stats` y retorna un resumen consolidado.
- **Detección de artículo**: Si la consulta hace referencia a un número de artículo en particular, realiza una consulta exacta en la base de datos. En caso de no existir un match exacto, recurre a la búsqueda semántica e inserta una advertencia.
- **Filtro de documentos**: Filtra dinámicamente las consultas según palabras clave asociadas a los nombres de los documentos (ej: "disciplinario" o "académico").
- **Búsqueda semántica**: Si no se detectan patrones especiales, calcula el embedding de la pregunta y busca en PostgreSQL los 5 fragmentos más cercanos conceptualmente.

> [!NOTE]
> **Sobre `tools.py`**: Aunque el archivo `tools.py` está presente en el código, **no se utiliza** ni se hace uso del protocolo de herramientas (`fastMCP` / Tool-calling del LLM). En su lugar, se implementó `retrieval.py` para inyectar contexto de manera proactiva en cada turno. Esto soluciona los problemas de inconsistencia presentados por modelos como `qwen2.5:7b` al decidir cuándo y cómo ejecutar llamadas de herramientas de manera autónoma, logrando búsquedas de conteo y filtros más robustos.

### `chat.py`
Módulo de la interfaz de usuario en terminal. Sus funciones principales son:
- Proveer el bucle de interacción CLI (*read-eval-print loop*).
- Gestionar el prompt del sistema (`SYSTEM_PROMPT`) que restringe al LLM a responder exclusivamente en español y apoyarse únicamente en el contexto aportado.
- Administrar el historial de chat de forma que no se acumule contexto redundante, evitando saturar la memoria o la ventana de contexto del LLM.

### `tools.py`
Archivo de herramientas alternativo (inactivo en producción). Diseñado originalmente para habilitar Ollama Tool Calling definiendo el esquema JSON de la función `search_knowledge_base`.

---

## 3. Glosario de Términos

- **RAG (Generación Aumentada por Recuperación / Retrieval-Augmented Generation)**: Framework que optimiza la salida de un modelo de lenguaje grande (LLM) al consultar una base de datos externa confiable antes de formular la respuesta.
- **Embedding (Incrustación)**: Vector de números que representa el significado semántico de un texto. Permite medir matemáticamente la similitud conceptual entre fragmentos de información.
- **pgvector**: Extensión de PostgreSQL que añade soporte para almacenar y consultar vectores de alta dimensión directamente mediante SQL.
- **HNSW (Hierarchical Navigable Small World)**: Índice espacial multidimensional usado para la búsqueda aproximada del vecino más cercano. Optimiza drásticamente el tiempo de respuesta de búsquedas vectoriales.
- **Chunking**: Proceso de particionar un texto largo en segmentos coherentes y delimitados para que puedan ser asimilados de forma óptima por el modelo de embeddings y el LLM.
- **Ollama**: Plataforma ligera y de código abierto que facilita el despliegue local de modelos de lenguaje grandes (LLMs) y embeddings sin depender de nubes de terceros.
- **Ventana de Contexto (Context Window)**: Cantidad máxima de datos de texto (medida en tokens) que un modelo de IA puede procesar simultáneamente en una sola llamada de inferencia.
- **MarkItDown**: Biblioteca desarrollada por Microsoft enfocada en transformar archivos de formatos diversos (incluidos PDFs) a Markdown limpio y legible por computadoras.
- **Alucinación**: Fenómeno en el cual un modelo de lenguaje genera información plausible pero fácticamente incorrecta o no presente en sus datos de entrenamiento u origen.
