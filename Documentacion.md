# RAG Local para Consulta de Reglamentos

* **Nota**: La documentación fue escrita y en gran parte generada con ayuda de la IA agéntica Gemini 3.5 Flash

Este documento proporciona una visión general y técnica del proyecto de consulta de reglamentos académicos y disciplinarios implementado mediante RAG (Generación Aumentada por Recuperación) local. *

---

## 1. Resumen General del Proyecto

### ¿En qué consiste?
El proyecto es un sistema de **Generación Aumentada por Recuperación (RAG)** local diseñado para responder consultas sobre reglamentos institucionales (académicos y disciplinarios). A diferencia de los enfoques RAG convencionales que dependen únicamente de la búsqueda semántica —la cual suele fallar o confundirse al buscar números de artículos específicos, capítulos concretos o realizar conteos—, este sistema implementa una arquitectura de recuperación avanzada que combina:
1. **Búsqueda Híbrida (Vectorial + FTS)**: Combina la búsqueda por significado (vectorial) con la búsqueda de palabras clave tradicional (Full-Text Search en español) utilizando el algoritmo de fusión **Reciprocal Rank Fusion (RRF)** para ordenar los resultados de forma más óptima.
2. **Búsqueda por Coincidencia Exacta**: Búsqueda indexada en base de datos para números de artículos y capítulos específicos, garantizando respuestas exactas para consultas puntuales.
3. **Estadísticas Estructurales**: Consultas estructuradas de SQL a través de un registro jerárquico de documentos para resolver de manera determinista preguntas de agregación o conteo de secciones, capítulos, artículos o anexos.

### ¿Cómo funciona?
El ciclo de vida del dato consta de las siguientes fases:
1. **Conversión e Ingesta**: Los archivos PDF depositados en `rawDocuments/` se analizan mediante su hash SHA256. Si sufrieron modificaciones o son nuevos, se procesan usando la herramienta de Microsoft `MarkItDown` para convertirlos a Markdown estructurado y almacenarlos en `processedDocuments/`.
2. **Segmentación y Extracción de Estructura**: El documento Markdown se divide en secciones utilizando una estrategia en cascada (Encabezados, etiquetas de Artículo, o agrupación de párrafos de hasta 1500 caracteres). Al mismo tiempo, se extrae y normaliza la estructura del documento (identificando artículos, capítulos con números romanos, secciones y anexos) y se guarda en una tabla relacional dedicada.
3. **Vectorización y Almacenamiento**: Cada fragmento se envía a Ollama para calcular su vector utilizando el modelo `nomic-embed-text:latest` el cual dispone de 768 dimensiones. Los vectores, el texto limpio, el título de la sección, el número de capítulo y de artículo se guardan en PostgreSQL. La base de datos actualiza automáticamente una columna para búsqueda de texto completo en español.
4. **Recuperación y Chat**: Al iniciar la interfaz en `chat.py`, las preguntas no pasan directamente al LLM. Primero, la lógica en `retrieval.py` intercepta la consulta y genera el contexto idóneo:
   - Si detecta palabras clave como "disciplinario" o "académico", filtra la búsqueda únicamente a ese documento.
   - Si pide un conteo exacto de artículos o capítulos, consulta la tabla de estructura física. Si el conteo es temático (ej. cuántos artículos hablan de faltas), explora fragmentos de búsqueda híbrida y deduce los números de artículos únicos implicados.
   - Si se busca un artículo o capítulo exacto, realiza una consulta determinista en la base de datos con fallback a búsqueda híbrida si no existe.
   - Para el resto de preguntas libres, ejecuta la búsqueda híbrida (RRF).
   Este contexto enriquecido se inyecta en el prompt enviado al LLM local (Ollama) junto con instrucciones estrictas de mantener las respuestas breves y únicamente en español.

### Tecnologías Utilizadas
- **Python 3**: Lenguaje base para los scripts de ingesta, recuperación y conversación.
- **PostgreSQL 17 + pgvector**: Base de datos relacional y vectorial. Utiliza el índice `HNSW` (Hierarchical Navigable Small World) para acelerar búsquedas vectoriales, índices B-Tree estándar para números de artículos y capítulos, y un índice GIN sobre un vector autogenerado (`tsvector`) en español para la búsqueda por texto completo (FTS).
- **Docker & Docker Compose**: Automatiza el despliegue del motor PostgreSQL en un entorno aislado y reproducible.
- **Ollama**: Servidor local para ejecutar y consumir modelos de IA de forma privada:
  - **Llama de Chat**: `gemma3:4b` por defecto (o el configurado en `.env`).
  - **Lector de Embeddings**: `nomic-embed-text:latest`.
- **Microsoft MarkItDown**: Extractor de texto que simplifica la conversión de PDFs a Markdown estructurado.

### ¿Cómo se prueba?
El proyecto se valida a través de ejecuciones de sus scripts clave:
1. **Prueba de Ingesta**: Ejecutando `python -m src.ingest`. el script revisa si hay nuevos archivos PDF o cambios en su contenido, y si los hay, crea su equivalente Markdown, extrayendo la estructura jerárquica y cargando correctamente la base de datos.
2. **Prueba de Conversación (Chat)**: Ejecutando `python -m src.chat`. Este es para comprobar la correcta lógica de recuperación, donde se pueden evaluar los siguientes comportamientos:
   - **Preguntas de conteo simple**: *"¿Cuántos artículos tiene el reglamento disciplinario?"* o *"¿Cuántos capítulos hay?"* (deberá activar la función de conteo de la estructura física en la base de datos).
   - **Preguntas de conteo temático**: *"¿Cuántos artículos hablan sobre sanciones?"* (deberá buscar fragmentos semánticamente relevantes e identificar los artículos únicos relacionados).
   - **Preguntas de artículo o capítulo exacto**: *"¿Qué dice el artículo 15?"* o *"¿Qué dice el capítulo 3 del reglamento académico?"* (deberá retornar exactamente la sección indicada, sin variaciones semánticas erróneas).
   - **Preguntas libres**: Consultas generales sobre faltas o deberes para comprobar el motor de búsqueda híbrida (RRF).

---

## 2. Funcionamiento y Estructura del Código (`./src`)

La lógica central y los módulos del proyecto se encuentran agrupados bajo el directorio `./src`. A continuación, se detalla la responsabilidad de cada archivo:

### `__init__.py`
Permite que Python reconozca el directorio `src` como un paquete del sistema, facilitando las importaciones relativas y absolutas entre módulos.

### `config.py`
Módulo de configuración centralizada. Se encarga de:
- Cargar variables del entorno desde el archivo `.env`.
- Resolver las rutas del proyecto a nivel de sistema operativo (`rawDocuments/` y `processedDocuments/`).
- Centralizar las credenciales de base de datos (DSN) y configuraciones de Ollama, estableciendo por defecto el modelo de chat `gemma3:4b` y el modelo de embeddings `nomic-embed-text:latest` con una ventana de contexto de 4096 tokens.

### `db.py`
Capa de persistencia y comunicación con PostgreSQL mediante el driver `psycopg` (versión 3) y `pgvector`. Sus funciones clave son:
- `get_connection()`: Abre conexiones y registra el adaptador vectorial.
- `document_already_processed()` y `register_document()`: Previenen el reprocesamiento de archivos inalterados usando hashes SHA256.
- `insert_chunk()` y `delete_chunks_of_document()`: Gestión del ciclo de vida de los fragmentos en la tabla `chunks`.
- `search_similar_chunks()`: Ejecuta una **búsqueda híbrida** que combina búsqueda vectorial de coseno con búsqueda por texto completo (`websearch_to_tsquery` y `ts_rank_cd`), unificando los rankings mediante **Reciprocal Rank Fusion (RRF)**.
- `get_chunks_by_article_number()` y `get_chunks_by_chapter_number()`: Recuperan de forma exacta e indexada secciones basadas en su artículo o capítulo.
- `insert_structure_entries()`, `delete_structure_of_document()` y `get_structure_stats()`: Permiten guardar y consultar la tabla `document_structure` para calcular conteos y estadísticas estructuradas exactas por tipo de entidad.

### `ingest.py`
Pipeline de procesamiento de documentos. Prepara la base de datos:
1. Escanea `rawDocuments/` y calcula el hash de los archivos.
2. Convierte archivos PDF a Markdown con `MarkItDown`.
3. Extrae la estructura de entidades del documento (`extract_document_structure`) identificando artículos, capítulos (convirtiendo números romanos a enteros), secciones y anexos.
4. Divide el contenido en fragmentos utilizando la estrategia en cascada (`split_by_sections`).
5. Genera embeddings llamando a Ollama y almacena tanto los fragmentos (`chunks`) con sus metadatos (título, contenido, vector, artículo, capítulo) como su estructura relacional (`document_structure`) en PostgreSQL.

### `retrieval.py`
Constituye la inteligencia de recuperación del RAG. Implementa `build_context`, una función que analiza la pregunta del usuario mediante expresiones regulares y lógica inteligente antes de consultar el LLM:
- **Filtro de documentos**: Identifica si el usuario pregunta específicamente por un reglamento ("disciplinario" o "académico") para restringir la consulta.
- **Detección de conteo**: 
  - Si es un *conteo estructural simple* (ej. "¿Cuántos artículos hay?"), consulta la base de datos estructural con `get_structure_stats`.
  - Si es un *conteo temático* (ej. "¿Cuántos artículos tratan de sanciones?"), utiliza `_thematic_count_and_search` analizando un conjunto amplio de 40 fragmentos recuperados para deducir qué artículos/capítulos únicos corresponden al tema.
- **Detección de artículo y capítulo**: Si la consulta hace referencia a un número de artículo o capítulo exacto, realiza la consulta precisa de base de datos. Si no existe, recurre a la búsqueda híbrida y añade una advertencia de fallback.
- **Búsqueda general**: Si no se detectan patrones especiales, calcula el embedding de la pregunta y busca en PostgreSQL utilizando el motor de búsqueda híbrida (RRF).

> [!NOTE]
> **Sobre `tools.py`**: Aunque el archivo `tools.py` está presente en el código, **no se utiliza** en producción ni se hace uso del protocolo de llamadas a herramientas (Tool-calling). En su lugar, se implementó la lógica directa en `retrieval.py` para enrutar las consultas y generar el contexto. Esto soluciona los problemas de inconsistencia presentados por modelos locales al decidir cuándo y cómo ejecutar llamadas de herramientas de manera autónoma, garantizando respuestas de conteo y búsquedas de artículos mucho más estables y deterministas.

### `chat.py`
Módulo de la interfaz de usuario en terminal. Se encarga de:
- Proveer el bucle de interacción CLI (*read-eval-print loop*).
- Gestionar el prompt del sistema (`SYSTEM_PROMPT`) que restringe al LLM a responder obligatoriamente en español, basándose exclusivamente en el contexto recuperado, sin alucinar y en lenguaje natural sin formato markdown.
- Administrar el historial de chat limitándolo únicamente a la última interacción (`[-2:]`) para prevenir el desbordamiento de la ventana de contexto y mantener un procesamiento rápido.
- Invocar el chat del modelo de Ollama con la opción `think=False` para evitar la generación de tokens de razonamiento visibles o innecesarios.

### `tools.py`
Archivo de herramientas alternativo (inactivo en producción). Contiene la definición del esquema JSON de `search_knowledge_base` compatible con Ollama Tool Calling, diseñado originalmente como alternativa para habilitar llamadas a funciones externas.

---

## 3. Glosario de Términos

- **RAG (Generación Aumentada por Recuperación / Retrieval-Augmented Generation)**: Framework que optimiza la salida de un modelo de lenguaje grande (LLM) al consultar una base de datos externa confiable antes de formular la respuesta.
- **Embedding (Incrustación)**: Vector de números que representa el significado semántico de un texto, permitiendo medir matemáticamente la similitud conceptual entre fragmentos de información.
- **pgvector**: Extensión de PostgreSQL que añade soporte para almacenar y consultar vectores de alta dimensión directamente mediante SQL.
- **HNSW (Hierarchical Navigable Small World)**: Índice espacial multidimensional usado para la búsqueda aproximada del vecino más cercano en bases de datos vectoriales.
- **Chunking**: Proceso de particionar un texto largo en fragmentos coherentes y delimitados para que puedan ser asimilados de forma óptima por el modelo de embeddings y el LLM.
- **Ollama**: Plataforma ligera y de código abierto que facilita el despliegue local de modelos de lenguaje grandes (LLMs) y embeddings de forma local.
- **Ventana de Contexto (Context Window)**: Cantidad máxima de datos de texto (medida en tokens) que un modelo de IA puede procesar simultáneamente en una sola llamada de inferencia.
- **MarkItDown**: Biblioteca de Microsoft enfocada en transformar archivos de formatos diversos (incluidos PDFs) a Markdown limpio y legible.
- **Alucinación**: Fenómeno en el cual un modelo de lenguaje genera información plausible pero fácticamente incorrecta o ausente en sus fuentes.
- **Búsqueda Híbrida (Hybrid Search)**: Técnica de recuperación de información que combina la búsqueda semántica basada en vectores con la búsqueda por palabras clave clásica (Full-Text Search).
- **RRF (Reciprocal Rank Fusion)**: Algoritmo de combinación que toma los resultados ordenados de múltiples sistemas de búsqueda y calcula un nuevo ranking unificado basándose en la posición recíproca de cada resultado en las listas originales.
- **FTS (Full-Text Search / Búsqueda de Texto Completo)**: Sistema de indexación y búsqueda que analiza lingüísticamente un texto para buscar palabras o términos específicos.
- **document_structure**: Tabla relacional implementada para registrar y indexar de forma precisa la jerarquía estructural (artículos, capítulos, secciones y anexos) extraída de los reglamentos.
