"""
=============================================================================
L6.py - Sistema de Agente IA con Memoria Persistente (MongoDB Checkpointer)
=============================================================================

Este módulo extiende el agente de L5 agregando MEMORIA PERSISTENTE usando
MongoDBSaver como checkpointer. Esto permite que el agente recuerde
conversaciones anteriores entre ejecuciones del programa.

DIFERENCIA CLAVE CON L5:
------------------------
L5: Sin memoria - cada ejecución es independiente
L6: Con memoria - el agente recuerda conversaciones anteriores usando thread_id

ARQUITECTURA DEL SISTEMA:
-------------------------
1. El usuario envía una consulta con un thread_id
2. El checkpointer carga el historial de ese thread desde MongoDB
3. El agente procesa la consulta con contexto del historial
4. Cada paso se guarda automáticamente en MongoDB
5. En futuras ejecuciones, el historial se restaura

PERSISTENCIA DE ESTADO:
-----------------------
MongoDBSaver guarda checkpoints del estado del grafo en MongoDB.
Cada thread_id tiene su propio historial de conversación.
Los checkpoints incluyen: mejsajes, estado del grafo, metadatos.

COMPONENTES PRINCIPALES:
------------------------
- MongoDBSaver: Checkpointer que persiste estado en MongoDB
- thread_id: Identificador único para cada conversación
- Todas las demás funciones de L5 (búsqueda vectorial, herramientas, etc.)

CASOS DE USO:
-------------
- Chatbots que recuerdan usuarios entre sesiones
- Agentes multi-turno donde el contexto es importante
- Sistemas donde "¿Qué te pregunté antes?" es una consulta válida

DEPENDENCIAS:
-------------
- langgraph.checkpoint.mongodb: MongoDBSaver para persistencia
- pymongo: Driver de MongoDB para Python
- boto3, langchain_aws: Para modelos AWS Bedrock
- langgraph, langchain_core: Framework del agente

AUTOR: Curso MongoDB AI Agents
VERSIÓN: 1.0 (Lección 6 - Memoria)
"""

# =============================================================================
# SECCIÓN 1: IMPORTACIONES Y CONFIGURACIÓN DEL ENTORNO
# =============================================================================

import sys  # Módulo sys: proporciona acceso a variables del intérprete Python
import os   # Módulo os: proporciona funciones para interactuar con el sistema operativo

# -----------------------------------------------------------------------------
# Configuración del Path del Sistema
# -----------------------------------------------------------------------------
# Esta línea permite que Python encuentre módulos en el directorio padre.
# __file__ es la ruta absoluta de este archivo Python
# os.path.dirname() obtiene el directorio que lo contiene
# os.path.join(..., '../../') sube dos niveles en la jerarquía
# os.path.abspath() convierte a ruta absoluta
# sys.path.append() agrega la ruta al path de búsqueda de módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# -----------------------------------------------------------------------------
# Importación de Módulos de Configuración
# -----------------------------------------------------------------------------
# key_param: Módulo personalizado con credenciales (mongodb_uri, api_keys, etc.)
import key_param

# -----------------------------------------------------------------------------
# Importación del Cliente MongoDB
# -----------------------------------------------------------------------------
# MongoClient: Clase principal de PyMongo para conectarse a MongoDB Atlas
# Se usa tanto para las colecciones de datos como para el checkpointer
from pymongo import MongoClient

# -----------------------------------------------------------------------------
# Importación del Decorador de Herramientas
# -----------------------------------------------------------------------------
# @tool: Decorador que convierte funciones Python en herramientas invocables
# por el LLM. El docstring de la función es crucial porque el LLM lo lee
# para decidir cuándo y cómo usar cada herramienta.
from langchain_core.tools import tool

# -----------------------------------------------------------------------------
# Importación de Tipos para Type Hints
# -----------------------------------------------------------------------------
# List: Para tipar listas (ej: List[float] = lista de números flotantes)
# Annotated: Permite agregar metadatos a tipos (usado con reducers)
from typing import List
from typing import Annotated

# -----------------------------------------------------------------------------
# Importación del Reducer de Mensajes
# -----------------------------------------------------------------------------
# add_messages: Función reducer que define cómo actualizar el campo messages.
# En lugar de reemplazar la lista, AGREGA nuevos mensajes preservando el historial.
# Esto es fundamental para mantener el contexto de la conversación.
from langgraph.graph.message import add_messages

# -----------------------------------------------------------------------------
# Importación de TypedDict
# -----------------------------------------------------------------------------
# TypedDict: Permite crear diccionarios con estructura fija y tipos definidos.
# Es la base para definir el esquema del estado del grafo.
from typing_extensions import TypedDict

# -----------------------------------------------------------------------------
# Importación de Templates de Prompt
# -----------------------------------------------------------------------------
# ChatPromptTemplate: Crea templates de prompts con variables dinámicas
# MessagesPlaceholder: Marca dónde insertar la lista de mensajes
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# -----------------------------------------------------------------------------
# Importación de Tipos de Mensajes
# -----------------------------------------------------------------------------
# ToolMessage: Representa el resultado de ejecutar una herramienta.
# Contiene el contenido y el tool_call_id para correlación.
from langchain_core.messages import ToolMessage

# -----------------------------------------------------------------------------
# Importación de Componentes del Grafo
# -----------------------------------------------------------------------------
# END: Nodo final del grafo (constante especial)
# StateGraph: Clase para construir grafos de estados
# START: Nodo inicial del grafo (constante especial)
from langgraph.graph import END, StateGraph, START

# -----------------------------------------------------------------------------
# Importación del Checkpointer de MongoDB - NUEVO EN L6
# -----------------------------------------------------------------------------
# MongoDBSaver: Implementación de Checkpointer que persiste en MongoDB.
# Guarda el estado del grafo después de cada nodo, permitiendo:
# 1. Continuidad de conversación entre ejecuciones
# 2. Recuperación del historial completo por thread_id
# 3. Tolerancia a fallos (se puede reanudar desde el último checkpoint)
from langgraph.checkpoint.mongodb import MongoDBSaver

# -----------------------------------------------------------------------------
# Importación de Utilidades del Sistema
# -----------------------------------------------------------------------------
# os: Ya importado, pero se re-importa para claridad
# getpass: Para solicitar credenciales de forma segura (sin eco en pantalla)
import os, getpass

# -----------------------------------------------------------------------------
# Importación del Cliente AWS Bedrock
# -----------------------------------------------------------------------------
# ChatBedrockConverse: Wrapper de LangChain para modelos de AWS Bedrock
# con la API Converse que soporta tool calling nativo.
from langchain_aws import ChatBedrockConverse

# -----------------------------------------------------------------------------
# Importación del SDK de AWS
# -----------------------------------------------------------------------------
# boto3: SDK oficial de AWS para Python
# Se usa para invocar directamente el modelo de embeddings Titan
import boto3

# -----------------------------------------------------------------------------
# Importación de JSON
# -----------------------------------------------------------------------------
# json: Módulo estándar para serialización/deserialización JSON
import json


# =============================================================================
# SECCIÓN 2: FUNCIONES AUXILIARES
# =============================================================================

def _set_env(var: str):
    """
    Establece una variable de entorno de forma interactiva si no existe.
    
    Esta función auxiliar verifica si una variable de entorno está definida.
    Si no lo está, solicita al usuario que ingrese el valor usando getpass,
    que oculta la entrada (útil para tokens y contraseñas).
    
    FLUJO DE EJECUCIÓN:
    -------------------
    1. os.environ.get(var) intenta obtener la variable
    2. Si retorna None/vacío (evaluado como False), entra al if
    3. getpass.getpass() muestra el prompt y lee sin eco
    4. El valor se almacena en os.environ para uso posterior
    
    Args:
        var (str): Nombre de la variable de entorno.
                   Ejemplos: "AWS_ACCESS_KEY_ID", "MONGODB_URI"
    
    Returns:
        None: Modifica os.environ como side effect.
    
    Ejemplo:
        _set_env("API_KEY")
        # Si API_KEY no existe, muestra: "API_KEY: " y espera input
    """
    # os.environ.get() retorna None si la variable no existe
    if not os.environ.get(var):
        # getpass.getpass() solicita entrada sin mostrarla en pantalla
        os.environ[var] = getpass.getpass(f"{var}: ")


# =============================================================================
# SECCIÓN 3: CONFIGURACIÓN DE MODELOS LLM
# =============================================================================
# 
# Se definen múltiples configuraciones de LLM para diferentes casos de uso.
# Cada modelo tiene diferentes características de velocidad, costo y capacidad.
# 
# PARÁMETROS DE CONFIGURACIÓN:
# - model_id: Identificador único del modelo en AWS Bedrock
# - region_name: Región de AWS (us-east-1 = Virginia del Norte)
# - temperature: Controla creatividad (0=determinista, 1=muy aleatorio)
# - max_tokens: Límite máximo de tokens en la respuesta generada
# - top_p: Nucleus sampling - qué proporción de tokens considerar
# =============================================================================

# -----------------------------------------------------------------------------
# Modelo 1: DeepSeek R1 - Razonamiento Complejo
# -----------------------------------------------------------------------------
# DeepSeek R1 está diseñado para tareas de razonamiento profundo.
# Características:
# - Produce "cadenas de pensamiento" detalladas
# - Mejor en matemáticas, lógica y planificación
# - Más lento pero más preciso en tareas complejas
llm_ds = ChatBedrockConverse(
    model_id="us.deepseek.r1-v1:0",  # ID validado en marketplace AWS
    region_name="us-east-1",          # Región de AWS
    temperature=0.6,                  # Moderadamente creativo
    max_tokens=8192,                  # Alto límite para respuestas largas
    top_p=0.95                        # Considera el 95% superior de tokens
)

# -----------------------------------------------------------------------------
# Modelo 2: Llama 4 Scout - Rápido y Eficiente
# -----------------------------------------------------------------------------
# Scout es una versión ligera optimizada para velocidad.
# Características:
# - Inferencia rápida con baja latencia
# - Temperature muy baja para respuestas consistentes
# - Ideal para tareas simples donde velocidad importa
llm_llama = ChatBedrockConverse(
    model_id="us.meta.llama4-scout-17b-instruct-v1:0",
    region_name="us-east-1",
    temperature=0.1,  # Muy determinista
    max_tokens=2048,
    top_p=0.9
)

# -----------------------------------------------------------------------------
# Modelo 3: Llama 4 Maverick - Balance Calidad/Velocidad
# -----------------------------------------------------------------------------
# Maverick ofrece un equilibrio entre calidad y rendimiento.
# Características:
# - Buen rendimiento general
# - Temperature moderada para cierta creatividad
# - Versátil para diferentes tipos de tareas
llm_maverick = ChatBedrockConverse(
    model_id="us.meta.llama4-maverick-17b-instruct-v1:0",
    region_name="us-east-1",
    temperature=0.5,  # Balance creatividad/coherencia
    max_tokens=2048,
    top_p=0.9
)

# -----------------------------------------------------------------------------
# Modelo 4: Amazon Nova Lite - Económico
# -----------------------------------------------------------------------------
# Nova Lite es el modelo liviano de Amazon, optimizado para costo.
# Características:
# - Menor costo por token
# - Bueno para prototipado y desarrollo
# - Rendimiento aceptable para tareas básicas
llm_nova = ChatBedrockConverse(
    model_id="amazon.nova-lite-v1:0",
    region_name="us-east-1",
    temperature=0.5,
    max_tokens=2048,
    top_p=0.9
)


# =============================================================================
# SECCIÓN 4: INICIALIZACIÓN DE MONGODB
# =============================================================================

def init_mongodb():
    """
    Inicializa la conexión a MongoDB y retorna cliente y colecciones.
    
    Esta función es crítica porque:
    1. Establece la conexión principal a MongoDB Atlas
    2. Proporciona acceso a las colecciones de documentos
    3. El cliente también se usa para el MongoDBSaver (checkpointer)
    
    ESQUEMA DE LA BASE DE DATOS:
    ----------------------------
    Database: ai_agents
    │
    ├── Collection: chunked_docs
    │   └── Documentos fragmentados con embeddings para búsqueda vectorial
    │   └── Campos: embedding (vector 1024D), body (texto), ...
    │   └── Índice: vector_index (para $vectorSearch)
    │
    ├── Collection: full_docs
    │   └── Documentos completos sin fragmentar
    │   └── Campos: title (único), body (texto completo), ...
    │
    └── Collections de LangGraph (creadas automáticamente por MongoDBSaver):
        └── checkpoints, checkpoint_writes, etc.
    
    NOTA SOBRE EL CLIENTE:
    ----------------------
    El mismo MongoClient sirve tanto para las colecciones de datos
    como para el MongoDBSaver. Esto es eficiente porque PyMongo
    maneja internamente un pool de conexiones.
    
    Returns:
        tuple: (mongodb_client, vs_collection, full_collection)
            - mongodb_client: Cliente MongoDB para el checkpointer
            - vs_collection: Colección con documentos embedidos
            - full_collection: Colección con documentos completos
    """
    # Crear cliente MongoDB usando el URI de conexión de key_param
    # El URI típicamente es: mongodb+srv://user:pass@cluster.mongodb.net
    mongodb_client = MongoClient(key_param.mongodb_uri)
    
    # Nombre de la base de datos que contiene la documentación
    DB_NAME = "ai_agents"
    
    # Colección con fragmentos de documentos y sus embeddings
    # Esta colección tiene el índice vectorial "vector_index"
    vs_collection = mongodb_client[DB_NAME]["chunked_docs"]
    
    # Colección con documentos completos (para resúmenes)
    # Cada documento tiene un título único usado para búsqueda exacta
    full_collection = mongodb_client[DB_NAME]["full_docs"]
    
    return mongodb_client, vs_collection, full_collection


# =============================================================================
# SECCIÓN 5: DEFINICIÓN DEL ESTADO DEL GRAFO
# =============================================================================

class GraphState(TypedDict):
    """
    Define la estructura del estado que fluye y se persiste en el grafo.
    
    En L6, el estado no solo fluye entre nodos durante la ejecución,
    sino que también se PERSISTE en MongoDB mediante el checkpointer.
    Esto permite recuperar el estado completo en futuras ejecuciones.
    
    PERSISTENCIA DEL ESTADO:
    ------------------------
    Cuando MongoDBSaver está activo:
    1. Después de cada nodo, el estado se guarda en MongoDB
    2. El checkpoint incluye: messages, metadata, thread_id
    3. Al iniciar con un thread_id existente, se carga el último estado
    
    EL REDUCER add_messages:
    ------------------------
    El tipo Annotated[list, add_messages] usa un reducer que:
    - AGREGA nuevos mensajes en lugar de reemplazar
    - Preserva todo el historial de la conversación
    - Permite queries como "¿Qué me preguntaste antes?"
    
    TIPOS DE MENSAJES EN EL HISTORIAL:
    -----------------------------------
    - HumanMessage: Preguntas/input del usuario
    - AIMessage: Respuestas del asistente (pueden incluir tool_calls)
    - ToolMessage: Resultados de ejecutar herramientas
    
    Campos:
        messages: Lista de todos los mensajes de la conversación.
                  Se preserva entre ejecuciones usando el thread_id.
    """
    # Lista de mensajes con reducer add_messages
    # El reducer garantiza que los mensajes se acumulen, no se reemplacen
    messages: Annotated[list, add_messages]


# =============================================================================
# SECCIÓN 6: GENERACIÓN DE EMBEDDINGS
# =============================================================================

def generate_embedding(text: str) -> List[float]:
    """
    Genera un embedding vectorial para un texto usando AWS Titan.
    
    Los embeddings son representaciones numéricas del significado semántico.
    Permiten comparar la similitud entre textos usando distancia vectorial.
    
    PROCESO DE EMBEDDING:
    ---------------------
    Texto "¿Cómo hago backup en MongoDB?"
                    ↓
    Modelo Titan Embeddings v2
                    ↓
    Vector [0.12, -0.34, 0.56, ..., 0.78] (1024 dimensiones)
    
    VENTAJAS DE TITAN V2:
    ---------------------
    - Dimensionalidad configurable (256, 512, 1024)
    - Normalización integrada para similitud coseno
    - Sin necesidad de API key adicional (usa AWS credentials)
    - Optimizado para texto en inglés y español
    
    Args:
        text (str): Texto a convertir en embedding.
                    Puede ser una consulta del usuario o un documento.
    
    Returns:
        List[float]: Vector de 1024 dimensiones (números flotantes).
                     Vector normalizado (longitud = 1) para similitud coseno.
    
    Ejemplo:
        >>> embedding = generate_embedding("MongoDB backup best practices")
        >>> len(embedding)
        1024
        >>> type(embedding[0])
        <class 'float'>
    """
    # PASO 1: Crear cliente de Bedrock Runtime
    # bedrock-runtime es el servicio para inferencia de modelos
    # Las credenciales AWS se obtienen del entorno automáticamente
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )

    # PASO 2: Preparar el payload JSON para Titan
    # Titan v2 acepta estos parámetros:
    # - inputText: El texto a embeddear
    # - dimensions: Tamaño del vector (256/512/1024)
    # - normalize: Si normalizar a longitud unitaria
    body = json.dumps({
        "inputText": text,
        "dimensions": 1024,  # Máxima precisión semántica
        "normalize": True    # Para similitud coseno directa
    })

    # PASO 3: Invocar el modelo de embeddings
    # El modelo procesa el texto y genera el vector
    response = bedrock_runtime.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="application/json",
        body=body
    )

    # PASO 4: Extraer el embedding de la respuesta
    # La respuesta es un stream que debemos leer y parsear
    response_body = json.loads(response.get("body").read())
    embedding = response_body.get("embedding")
    
    return embedding


# =============================================================================
# SECCIÓN 7: HERRAMIENTAS DEL AGENTE
# =============================================================================
# 
# Las herramientas son funciones que el LLM puede invocar cuando necesita
# información externa. El decorador @tool las registra y expone al LLM.
# =============================================================================

@tool
def get_information_for_question_answering(user_query: str) -> str:
    """
    Busca documentos relevantes para responder preguntas usando búsqueda vectorial.
    
    CUÁNDO USA EL LLM ESTA HERRAMIENTA:
    -----------------------------------
    El LLM invoca esta herramienta cuando el usuario hace preguntas
    que requieren información factual sobre MongoDB:
    - "¿Cuáles son las mejores prácticas para backups?"
    - "¿Cómo configuro replicación en MongoDB?"
    - "¿Qué es el sharding y cuándo usarlo?"
    
    CÓMO FUNCIONA LA BÚSQUEDA VECTORIAL:
    ------------------------------------
    1. La pregunta se convierte en un embedding (vector 1024D)
    2. MongoDB busca documentos con embeddings similares
    3. La similitud se calcula usando distancia vectorial
    4. Los 5 más similares se retornan como contexto
    
    PIPELINE DE MONGODB:
    --------------------
    $vectorSearch → Encuentra documentos similares por embedding
    $project → Selecciona solo los campos necesarios (body, score)
    
    Args:
        user_query (str): La pregunta del usuario en lenguaje natural.
    
    Returns:
        str: Texto concatenado de los 5 documentos más relevantes,
             separados por doble salto de línea para claridad.
    """
    # Convertir la pregunta en un embedding para búsqueda vectorial
    query_embedding = generate_embedding(user_query)

    # Obtener la colección de búsqueda vectorial (índice 1 del tuple)
    vs_collection = init_mongodb()[1]
    
    # Pipeline de agregación para búsqueda vectorial
    pipeline = [
        # Etapa 1: Búsqueda por similitud vectorial
        {
            "$vectorSearch": {
                "index": "vector_index",        # Nombre del índice en Atlas
                "path": "embedding",            # Campo con los embeddings
                "queryVector": query_embedding, # Vector de la consulta
                "numCandidates": 150,           # Candidatos a evaluar
                "limit": 5,                     # Top 5 resultados finales
            }
        },
        # Etapa 2: Proyectar solo campos necesarios
        {
            "$project": {
                "_id": 0,                                    # Excluir _id
                "body": 1,                                   # Incluir texto
                "score": {"$meta": "vectorSearchScore"},     # Score de similitud
            }
        },
    ]
    
    # Ejecutar el pipeline y obtener resultados
    results = vs_collection.aggregate(pipeline)
    
    # Concatenar textos de todos los documentos encontrados
    context = "\n\n".join([doc.get("body") for doc in results])
    
    return context


@tool
def get_page_content_for_summarization(user_query: str) -> str:
    """
    Recupera el contenido completo de una página para resumir.
    
    CUÁNDO USA EL LLM ESTA HERRAMIENTA:
    -----------------------------------
    El LLM invoca esta herramienta cuando el usuario pide
    explícitamente un resumen de una página por su título:
    - "Dame un resumen de la página 'Create a MongoDB Deployment'"
    - "Summarize the 'Indexing Strategies' documentation"
    
    DIFERENCIA CON LA HERRAMIENTA QA:
    ----------------------------------
    - QA: Búsqueda semántica → fragmentos de múltiples documentos
    - Esta: Búsqueda exacta por título → documento completo
    
    Args:
        user_query (str): El título EXACTO de la página de documentación.
                          Debe coincidir exactamente con el campo 'title'.
    
    Returns:
        str: Contenido completo de la página (campo 'body'),
             o "Document not found" si no existe ese título.
    """
    # Obtener la colección de documentos completos (índice 2)
    full_collection = init_mongodb()[2]

    # Crear query de búsqueda por título exacto
    query = {"title": user_query}
    
    # Solo necesitamos el campo body
    projection = {"_id": 0, "body": 1}
    
    # Buscar el documento
    document = full_collection.find_one(query, projection)
    
    if document:
        return document["body"]
    else:
        return "Document not found"


# =============================================================================
# SECCIÓN 8: NODOS DEL GRAFO
# =============================================================================

def agent(state: GraphState, llm_with_tools) -> GraphState:
    """
    Nodo del agente: el "cerebro" que decide las acciones.
    
    Este nodo recibe el estado actual (incluyendo todo el historial
    de mensajes) y usa el LLM para decidir:
    1. Si necesita más información → genera tool_calls
    2. Si puede responder → genera la respuesta final
    
    CONTEXTO HISTÓRICO (NUEVO EN L6):
    ----------------------------------
    Gracias al checkpointer, el estado incluye mensajes de
    ejecuciones anteriores. El LLM puede ver:
    - Preguntas previas del usuario
    - Sus propias respuestas anteriores
    - Resultados de herramientas pasadas
    
    Esto permite preguntas como:
    - "¿Qué te pregunté antes?" → El LLM ve el historial
    - "Explícame más sobre eso" → El LLM tiene contexto
    
    Args:
        state (GraphState): Estado actual con historial de mensajes.
        llm_with_tools: Cadena prompt | LLM con herramientas vinculadas.
    
    Returns:
        GraphState: Diccionario con el nuevo mensaje del agente.
                    {"messages": [AIMessage(...)]}
    """
    # Extraer mensajes del estado (incluye historial si hay checkpoint)
    messages = state["messages"]
    
    # Invocar el LLM con todo el historial
    # El LLM analiza la conversación y genera su respuesta
    result = llm_with_tools.invoke(messages)
    
    # Retornar el nuevo mensaje para agregar al estado
    return {"messages": [result]}


def tool_node(state: GraphState, tools_by_name) -> GraphState:
    """
    Nodo de herramientas: ejecuta las herramientas solicitadas.
    
    Cuando el agente decide que necesita información, genera
    tool_calls. Este nodo:
    1. Lee las tool_calls del último mensaje del agente
    2. Ejecuta cada herramienta con sus argumentos
    3. Empaqueta resultados como ToolMessages
    
    ESTRUCTURA DE TOOL_CALL:
    ------------------------
    {
        "name": "get_information_for_question_answering",
        "args": {"user_query": "MongoDB backup practices"},
        "id": "call_abc123"  # ID único para correlación
    }
    
    Args:
        state (GraphState): Estado con el AIMessage que tiene tool_calls.
        tools_by_name (Dict): Mapeo {nombre: función} de herramientas.
    
    Returns:
        GraphState: Lista de ToolMessages con los resultados.
    """
    # Lista para acumular resultados
    result = []
    
    # Obtener tool_calls del último mensaje (AIMessage del agente)
    tool_calls = state["messages"][-1].tool_calls
    
    # Ejecutar cada herramienta solicitada
    for tool_call in tool_calls:
        # Buscar la función por nombre
        tool = tools_by_name[tool_call["name"]]
        
        # Invocar con los argumentos proporcionados
        observation = tool.invoke(tool_call["args"])
        
        # Crear ToolMessage con el resultado
        # El tool_call_id permite correlacionar solicitud y respuesta
        result.append(ToolMessage(
            content=observation,
            tool_call_id=tool_call["id"]
        ))
    
    return {"messages": result}


# =============================================================================
# SECCIÓN 9: FUNCIÓN DE ENRUTAMIENTO
# =============================================================================

def route_tools(state: GraphState):
    """
    Decide el siguiente paso basándose en el último mensaje del agente.
    
    Esta función examina si el agente solicitó herramientas:
    - Si hay tool_calls → ir al nodo "tools"
    - Si no hay tool_calls → ir a END (respuesta final)
    
    FLUJO DE DECISIÓN:
    ------------------
    Agent genera AIMessage
            ↓
    route_tools examina tool_calls
            ↓
    ┌───────┴───────┐
    ↓               ↓
    "tools"         END
    (hay calls)     (respuesta lista)
    
    Args:
        state (GraphState): Estado con el último mensaje del agente.
    
    Returns:
        str: "tools" si hay tool_calls, END si es respuesta final.
    
    Raises:
        ValueError: Si no hay mensajes en el estado.
    """
    # Obtener mensajes con default vacío
    messages = state.get("messages", [])
    
    # Verificar que hay mensajes
    if len(messages) > 0:
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    # Verificar si hay tool_calls pendientes
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    
    return END


# =============================================================================
# SECCIÓN 10: INICIALIZACIÓN DEL GRAFO CON PERSISTENCIA
# =============================================================================

def init_graph(llm_with_tools, tools_by_name, mongodb_client):
    """
    Inicializa el grafo de estados CON CHECKPOINTER para persistencia.
    
    DIFERENCIA CLAVE CON L5:
    ------------------------
    L5: graph.compile() sin argumentos
    L6: graph.compile(checkpointer=checkpointer) con MongoDBSaver
    
    EL CHECKPOINTER:
    ----------------
    MongoDBSaver guarda el estado del grafo en MongoDB después de cada nodo.
    Esto permite:
    1. Recuperar conversaciones por thread_id
    2. Continuar donde se dejó en ejecuciones futuras
    3. Mantener memoria a largo plazo del agente
    
    ARQUITECTURA DEL GRAFO:
    -----------------------
              START
                │
                ▼
            ┌─agent──┐
            │        │
      tool_calls?    no
            │        │
            ▼        ▼
          tools     END
            │
            └──────→agent (loop)
    
    PERSISTENCIA AUTOMÁTICA:
    ------------------------
    Con el checkpointer activo:
    - Cada transición de nodo guarda el estado
    - Se crea una colección "checkpoints" en MongoDB
    - Los checkpoints se indexan por thread_id
    
    Args:
        llm_with_tools: Cadena de procesamiento (prompt | LLM.bind_tools).
        tools_by_name (Dict): Mapeo de nombres a funciones de herramientas.
        mongodb_client (MongoClient): Cliente MongoDB para el checkpointer.
    
    Returns:
        CompiledGraph: Grafo compilado con persistencia habilitada.
    """
    # Crear instancia del grafo de estados
    graph = StateGraph(GraphState)
    
    # Agregar nodo del agente
    # Lambda para inyectar llm_with_tools como dependencia
    graph.add_node("agent", lambda state: agent(state, llm_with_tools))
    
    # Agregar nodo de herramientas
    # Lambda para inyectar tools_by_name como dependencia
    graph.add_node("tools", lambda state: tool_node(state, tools_by_name))
    
    # Arista de inicio: START → agent
    graph.add_edge(START, "agent")
    
    # Arista de retorno: tools → agent (loop)
    graph.add_edge("tools", "agent")
    
    # Aristas condicionales desde agent
    # route_tools decide: "tools" o END
    graph.add_conditional_edges(
        "agent",
        route_tools,
        {"tools": "tools", END: END}
    )
    
    # NUEVO EN L6: Crear el checkpointer de MongoDB
    # MongoDBSaver usa el cliente proporcionado para crear colecciones
    # de checkpoints en la base de datos
    checkpointer = MongoDBSaver(mongodb_client)
    
    # Compilar con checkpointer para habilitar persistencia
    return graph.compile(checkpointer=checkpointer)


# =============================================================================
# SECCIÓN 11: EJECUCIÓN DEL GRAFO CON THREAD_ID
# =============================================================================

def execute_graph(app, thread_id: str, user_input: str) -> None:
    """
    Ejecuta el grafo con persistencia usando thread_id.
    
    DIFERENCIA CLAVE CON L5:
    ------------------------
    L5: execute_graph(app, user_input) - sin thread_id
    L6: execute_graph(app, thread_id, user_input) - CON thread_id
    
    EL THREAD_ID:
    -------------
    El thread_id identifica una conversación específica.
    - Mismo thread_id = misma conversación (se acumula historial)
    - Diferente thread_id = nueva conversación independiente
    
    EJEMPLO DE USO:
    ---------------
    # Primera ejecución con thread "1"
    execute_graph(app, "1", "¿Qué es MongoDB?")
    # El agente responde y guarda en checkpoint
    
    # Segunda ejecución con mismo thread "1"
    execute_graph(app, "1", "¿Qué te pregunté antes?")
    # El agente carga el historial y puede ver la pregunta anterior
    
    CONFIG PARA CHECKPOINTER:
    -------------------------
    El diccionario {"configurable": {"thread_id": "..."}} indica
    al checkpointer qué hilo de conversación usar.
    
    Args:
        app: Grafo compilado con checkpointer.
        thread_id (str): Identificador único de la conversación.
                         Puede ser cualquier string (ej: user_id, session_id).
        user_input (str): La consulta del usuario.
    
    Returns:
        None: Imprime resultados a consola.
    """
    # Preparar el estado de entrada
    # La tupla ("user", texto) se convierte a HumanMessage
    input = {"messages": [("user", user_input)]}
    
    # NUEVO EN L6: Configuración con thread_id
    # Esta configuración le dice al checkpointer qué hilo usar
    config = {"configurable": {"thread_id": thread_id}}
    
    # Ejecutar en modo streaming con la configuración
    # El checkpointer carga el historial automáticamente
    for output in app.stream(input, config):
        for key, value in output.items():
            print(f"Node {key}:")
            print(value)
    
    print("---FINAL ANSWER---")
    print(value["messages"][-1].content)


# =============================================================================
# SECCIÓN 12: FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Función principal que demuestra la memoria del agente.
    
    DEMOSTRACIÓN DE MEMORIA:
    ------------------------
    Esta función ejecuta DOS consultas con el MISMO thread_id ("1"):
    
    1. Primera consulta: "What are some best practices for data backups?"
       - El agente responde normalmente
       - El estado se guarda en MongoDB
    
    2. Segunda consulta: "What did I just ask you?"
       - El checkpointer carga el historial del thread "1"
       - El agente VE la pregunta anterior en su contexto
       - Puede responder "Me preguntaste sobre backups"
    
    FLUJO COMPLETO:
    ---------------
    1. init_mongodb() → Cliente y colecciones
    2. Definir herramientas disponibles
    3. Seleccionar LLM (llm_nova por defecto)
    4. Crear prompt del sistema con instrucciones
    5. Vincular herramientas al LLM (bind_tools)
    6. Crear cadena de procesamiento (prompt | llm)
    7. Mapear herramientas por nombre
    8. Inicializar grafo CON checkpointer
    9. Ejecutar primera consulta (thread "1")
    10. Ejecutar segunda consulta (thread "1") → USA EL HISTORIAL
    
    Returns:
        None: Resultados impresos a consola.
    """
    # PASO 1: Inicializar MongoDB
    # El cliente se usa tanto para datos como para el checkpointer
    mongodb_client, vs_collection, full_collection = init_mongodb()
    
    # PASO 2: Definir las herramientas disponibles
    tools = [
        get_information_for_question_answering,  # Búsqueda vectorial
        get_page_content_for_summarization       # Recuperación por título
    ]
    
    # PASO 3: Seleccionar el LLM
    # llm = ChatOpenAI(openai_api_key=key_param.openai_api_key, temperature=0, model="gpt-4o")
    llm = llm_maverick  # Usando Nova Lite como default

    # PASO 4: Crear el prompt del sistema
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant."
                " You are provided with tools to answer questions and summarize technical documentation related to MongoDB."
                " Think step-by-step and use these tools to get the information required to answer the user query."
                " Do not re-run tools unless absolutely necessary."
                " If you are not able to get enough information using the tools, reply with I DON'T KNOW."
                " You have access to the following tools: {tool_names}."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    # PASO 5: Insertar nombres de herramientas en el prompt
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    
    # PASO 6: Vincular herramientas al LLM
    bind_tools = llm.bind_tools(tools)
    
    # PASO 7: Crear cadena de procesamiento
    llm_with_tools = prompt | bind_tools
    
    # PASO 8: Crear diccionario de herramientas
    tools_by_name = {tool.name: tool for tool in tools}
    
    # PASO 9: Inicializar grafo CON checkpointer
    # Pasamos mongodb_client para que MongoDBSaver lo use
    app = init_graph(llm_with_tools, tools_by_name, mongodb_client)
    
    # PASO 10: Primera ejecución con thread "1"
    # Esta consulta se responde y se guarda en el checkpoint
    execute_graph(app, "2", "What are some best practices for data backups in MongoDB?")
    
    # PASO 11: Segunda ejecución con MISMO thread "2"
    # El agente carga el historial y puede ver la pregunta anterior
    # Esto demuestra que la memoria está funcionando
    execute_graph(app, "2", "What did I just ask you?")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
# 
# Ejecuta main() cuando el archivo se ejecuta directamente.
# El if __name__ == "__main__" es una buena práctica pero main()
# se llama directamente aquí para simplicidad del curso.
# =============================================================================

main()