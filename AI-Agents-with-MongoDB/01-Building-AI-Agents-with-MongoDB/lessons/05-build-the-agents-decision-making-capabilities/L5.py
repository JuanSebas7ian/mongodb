"""
=============================================================================
L5.py - Sistema de Agente IA con Capacidades de Toma de Decisiones
=============================================================================

Este módulo implementa un agente de IA conversacional que puede responder
preguntas y resumir documentación técnica de MongoDB. Utiliza LangGraph para
orquestar el flujo de decisiones del agente y MongoDB Atlas para búsqueda
vectorial semántica.

ARQUITECTURA DEL SISTEMA:
-------------------------
1. El usuario envía una consulta
2. El agente (LLM) decide si necesita usar herramientas
3. Si necesita herramientas, las ejecuta y obtiene resultados
4. El agente genera una respuesta final basada en los resultados

COMPONENTES PRINCIPALES:
------------------------
- GraphState: Estado del grafo que mantiene el historial de mensajes
- generate_embedding(): Genera embeddings usando AWS Titan
- get_information_for_question_answering(): Herramienta de búsqueda vectorial
- get_page_content_for_summarization(): Herramienta de resumen de documentos
- agent(): Nodo del agente que invoca el LLM
- tool_node(): Nodo que ejecuta las herramientas
- route_tools(): Función de enrutamiento condicional
- init_graph(): Inicializa y compila el grafo de estados

FLUJO DE DATOS:
---------------
START -> agent -> (tools -> agent)* -> END

El asterisco (*) indica que el ciclo agent-tools puede repetirse
múltiples veces hasta que el agente decida que tiene suficiente
información para responder.

DEPENDENCIAS:
-------------
- boto3: SDK de AWS para invocar modelos de Bedrock
- pymongo: Driver de MongoDB para Python
- langgraph: Framework para construir agentes de IA
- langchain_aws: Integración de LangChain con AWS Bedrock

AUTOR: Curso MongoDB AI Agents
VERSIÓN: 1.0
"""

# =============================================================================
# SECCIÓN 1: IMPORTACIONES Y CONFIGURACIÓN DEL ENTORNO
# =============================================================================

import sys  # Módulo sys: proporciona acceso a variables del intérprete Python
import os   # Módulo os: proporciona funciones para interactuar con el sistema operativo

# -----------------------------------------------------------------------------
# Configuración del Path del Sistema
# -----------------------------------------------------------------------------
# Esta línea es crucial para que Python pueda encontrar módulos en el directorio padre.
# os.path.dirname(__file__) obtiene el directorio donde está este archivo
# os.path.join(..., '../../') sube dos niveles en la jerarquía de directorios
# os.path.abspath() convierte la ruta relativa en una ruta absoluta
# sys.path.append() agrega esta ruta a la lista de directorios donde Python busca módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# -----------------------------------------------------------------------------
# Importación de Módulos de Configuración
# -----------------------------------------------------------------------------
# key_param: Módulo personalizado que contiene las credenciales y URIs de conexión
# Este módulo típicamente contiene: mongodb_uri, openai_api_key, voyage_api_key, etc.
import key_param

# -----------------------------------------------------------------------------
# Importación de Cliente MongoDB
# -----------------------------------------------------------------------------
# MongoClient: Clase principal de PyMongo para conectarse a MongoDB
# Permite realizar operaciones CRUD y agregaciones en la base de datos
from pymongo import MongoClient

# -----------------------------------------------------------------------------
# Importación de Decorador de Herramientas LangChain
# -----------------------------------------------------------------------------
# @tool: Decorador que convierte una función Python en una herramienta
# que el LLM puede invocar. El LLM usa el docstring de la función
# para entender cuándo y cómo usar la herramienta.
# from langchain.agents import tool  # Versión antigua, deprecada
from langchain_core.tools import tool

# -----------------------------------------------------------------------------
# Importación de Tipos para Type Hints
# -----------------------------------------------------------------------------
# List: Tipo genérico para listas tipadas (ej: List[float] = lista de flotantes)
# Dict: Tipo genérico para diccionarios tipados
# Annotated: Permite agregar metadatos a los tipos (usado con reducers en LangGraph)
from typing import List, Dict
from typing import Annotated

# -----------------------------------------------------------------------------
# Importación de Reducer de Mensajes LangGraph
# -----------------------------------------------------------------------------
# add_messages: Función reducer que define CÓMO se actualizan los mensajes
# en el estado del grafo. En lugar de reemplazar la lista de mensajes,
# AGREGA nuevos mensajes a la lista existente, manteniendo el historial.
from langgraph.graph.message import add_messages

# -----------------------------------------------------------------------------
# Importación de TypedDict
# -----------------------------------------------------------------------------
# TypedDict: Permite crear diccionarios con estructura fija y tipos definidos
# Es fundamental para definir el esquema del estado del grafo
from typing_extensions import TypedDict

# -----------------------------------------------------------------------------
# Importación de Templates de Prompt
# -----------------------------------------------------------------------------
# ChatPromptTemplate: Clase para crear templates de prompts estructurados
# que pueden incluir variables dinámicas como {tool_names}
# MessagesPlaceholder: Marcador de posición para insertar una lista de mensajes
# en un punto específico del prompt (útil para historial de conversación)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# -----------------------------------------------------------------------------
# Importación de Tipos de Mensajes
# -----------------------------------------------------------------------------
# ToolMessage: Tipo de mensaje que representa el resultado de ejecutar
# una herramienta. Contiene el contenido del resultado y el ID de la
# llamada a la herramienta original para que el LLM pueda correlacionarlos.
from langchain_core.messages import ToolMessage

# -----------------------------------------------------------------------------
# Importación de Componentes del Grafo LangGraph
# -----------------------------------------------------------------------------
# END: Constante especial que representa el nodo final del grafo
# StateGraph: Clase principal para construir grafos de estado
# START: Constante especial que representa el nodo inicial del grafo
from langgraph.graph import END, StateGraph, START

# -----------------------------------------------------------------------------
# Importación de Utilidades del Sistema Operativo
# -----------------------------------------------------------------------------
# getpass: Módulo para solicitar contraseñas de forma segura (sin eco en pantalla)
import os, getpass

# -----------------------------------------------------------------------------
# Importación del Cliente AWS Bedrock
# -----------------------------------------------------------------------------
# ChatBedrockConverse: Wrapper de LangChain para modelos de AWS Bedrock
# que usan la API Converse. Soporta tool calling nativo.
from langchain_aws import ChatBedrockConverse

# -----------------------------------------------------------------------------
# Importación del SDK de AWS
# -----------------------------------------------------------------------------
# boto3: SDK oficial de AWS para Python. Se usa para invocar directamente
# el modelo de embeddings Titan (que no tiene wrapper de LangChain)
import boto3

# -----------------------------------------------------------------------------
# Importación de JSON
# -----------------------------------------------------------------------------
# json: Módulo estándar para serializar/deserializar JSON
# Necesario para preparar el payload de la API de Bedrock
import json


# =============================================================================
# SECCIÓN 2: FUNCIONES AUXILIARES
# =============================================================================

def _set_env(var: str):
    """
    Establece una variable de entorno de forma interactiva si no existe.
    
    Esta función es útil cuando se ejecuta el script en un entorno donde
    las variables de entorno no están preconfiguradas. Solicita al usuario
    que ingrese el valor de forma segura (sin mostrar lo que escribe).
    
    Args:
        var (str): Nombre de la variable de entorno a establecer
                   Ejemplo: "OPENAI_API_KEY", "AWS_ACCESS_KEY_ID"
    
    Comportamiento:
        - Si la variable YA existe en el entorno, no hace nada
        - Si NO existe, solicita el valor al usuario usando getpass
          (el texto ingresado no se muestra en pantalla por seguridad)
    
    Ejemplo de uso:
        _set_env("API_KEY")  # Si API_KEY no existe, pregunta: "API_KEY: "
    """
    # os.environ.get(var) retorna None si la variable no existe
    if not os.environ.get(var):
        # getpass.getpass() solicita entrada sin eco (para contraseñas/tokens)
        os.environ[var] = getpass.getpass(f"{var}: ")


# =============================================================================
# SECCIÓN 3: CONFIGURACIÓN DE MODELOS LLM
# =============================================================================
# 
# Esta sección define múltiples configuraciones de LLM para diferentes casos de uso.
# Cada modelo tiene características únicas en términos de velocidad, costo y capacidad.
# 
# PARÁMETROS COMUNES:
# - model_id: Identificador único del modelo en AWS Bedrock
# - region_name: Región de AWS donde se ejecuta el modelo (us-east-1 = Virginia)
# - temperature: Controla la aleatoriedad (0=determinista, 1=muy creativo)
# - max_tokens: Límite máximo de tokens en la respuesta
# - top_p: Nucleus sampling, probabilidad acumulada para selección de tokens
# =============================================================================

# -----------------------------------------------------------------------------
# Configuración 1: DeepSeek R1 - Modelo de Razonamiento Complejo
# -----------------------------------------------------------------------------
# DeepSeek R1 está optimizado para tareas que requieren razonamiento paso a paso.
# Ideal para: planificación, resolución de problemas matemáticos, análisis lógico.
# Nota: Este modelo puede ser más lento pero produce razonamientos más detallados.
llm_ds = ChatBedrockConverse(
    model_id="us.deepseek.r1-v1:0",  # ID del modelo en el marketplace de AWS
    region_name="us-east-1",          # Región de AWS
    temperature=0.6,                  # Balance entre creatividad y coherencia
    max_tokens=8192,                  # Límite alto para respuestas largas
    top_p=0.95                        # Considera el 95% de los tokens más probables
)

# -----------------------------------------------------------------------------
# Configuración 2: Llama 4 Scout - Modelo Ligero y Rápido
# -----------------------------------------------------------------------------
# Scout es una versión optimizada de Llama para inferencia rápida.
# Ideal para: respuestas rápidas, tareas simples, bajo costo.
# Temperature baja (0.1) para respuestas más deterministas y consistentes.
llm_llama = ChatBedrockConverse(
    model_id="us.meta.llama4-scout-17b-instruct-v1:0",  # Llama 4 Scout 17B
    region_name="us-east-1",
    temperature=0.1,  # Muy baja: respuestas predecibles y coherentes
    max_tokens=2048,  # Límite moderado
    top_p=0.9        # Ligeramente restrictivo
)

# -----------------------------------------------------------------------------
# Configuración 3: Llama 4 Maverick - Modelo Balanceado
# -----------------------------------------------------------------------------
# Maverick ofrece un balance entre calidad y velocidad.
# Ideal para: tareas generales, conversaciones, preguntas y respuestas.
llm_maverick = ChatBedrockConverse(
    model_id="us.meta.llama4-maverick-17b-instruct-v1:0",
    region_name="us-east-1",
    temperature=0.5,  # Balance entre creatividad y coherencia
    max_tokens=2048,
    top_p=0.9
)

# -----------------------------------------------------------------------------
# Configuración 4: Amazon Nova Lite - Modelo de Amazon
# -----------------------------------------------------------------------------
# Nova Lite es el modelo ligero de Amazon, optimizado para costo-eficiencia.
# Ideal para: prototipado, tareas simples, cuando el costo es una prioridad.
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
    Inicializa la conexión a MongoDB y retorna las colecciones necesarias.
    
    Esta función establece una conexión con MongoDB Atlas utilizando el URI
    de conexión almacenado en key_param. La conexión se usa para:
    1. Búsqueda vectorial en documentos fragmentados (chunked_docs)
    2. Recuperación de documentos completos para resumen (full_docs)
    
    ARQUITECTURA DE LA BASE DE DATOS:
    ---------------------------------
    Database: ai_agents
    └── Collection: chunked_docs (para búsqueda vectorial)
    │   └── Campos: embedding (vector), body (texto), ...
    └── Collection: full_docs (documentos completos)
        └── Campos: title, body, ...
    
    Returns:
        tuple: Una tupla de 3 elementos:
            - mongodb_client (MongoClient): Cliente de MongoDB para operaciones
            - vs_collection (Collection): Colección con embeddings vectoriales
            - full_collection (Collection): Colección con documentos completos
    
    Notas:
        - La conexión se mantiene abierta durante toda la ejecución
        - MongoDB Atlas maneja automáticamente el pool de conexiones
        - El índice vectorial 'vector_index' debe existir en chunked_docs
    """
    # Crear cliente MongoDB usando el URI de conexión
    # El URI típicamente tiene el formato: mongodb+srv://user:pass@cluster.mongodb.net/
    mongodb_client = MongoClient(key_param.mongodb_uri)
    
    # Nombre de la base de datos donde están almacenados los documentos
    DB_NAME = "ai_agents"
    
    # Colección para búsqueda vectorial (contiene documentos fragmentados con embeddings)
    # Esta colección tiene un índice vectorial llamado "vector_index"
    vs_collection = mongodb_client[DB_NAME]["chunked_docs"]
    
    # Colección con documentos completos (sin fragmentar, para resúmenes)
    # Cada documento tiene un título único que se usa para búsqueda exacta
    full_collection = mongodb_client[DB_NAME]["full_docs"]
    
    # Retornar cliente y ambas colecciones
    return mongodb_client, vs_collection, full_collection


# =============================================================================
# SECCIÓN 5: DEFINICIÓN DEL ESTADO DEL GRAFO
# =============================================================================

class GraphState(TypedDict):
    """
    Define la estructura del estado que fluye a través del grafo de decisiones.
    
    En LangGraph, el estado es un diccionario tipado que se pasa entre nodos.
    Cada nodo puede leer el estado actual y retornar actualizaciones.
    
    CONCEPTO CLAVE - REDUCERS:
    --------------------------
    El tipo Annotated[list, add_messages] indica que el campo 'messages'
    usa el reducer 'add_messages'. Esto significa que:
    
    - SIN reducer: state["messages"] = new_messages (REEMPLAZA todo)
    - CON add_messages: state["messages"] = existing + new_messages (AGREGA)
    
    Esto es fundamental para mantener el historial de conversación.
    
    Campos:
        messages (Annotated[list, add_messages]): 
            Lista de mensajes que representa el historial de conversación.
            Incluye mensajes del usuario, del asistente, y de las herramientas.
            
            Tipos de mensajes que puede contener:
            - HumanMessage: Mensajes del usuario
            - AIMessage: Respuestas del asistente (pueden incluir tool_calls)
            - ToolMessage: Resultados de ejecutar herramientas
    
    Ejemplo de flujo:
        Estado inicial: {"messages": [HumanMessage("Hola")]}
        Después del agente: {"messages": [HumanMessage, AIMessage(tool_calls=[...])]}
        Después de tools: {"messages": [HumanMessage, AIMessage, ToolMessage(resultado)]}
        Respuesta final: {"messages": [HumanMessage, AIMessage, ToolMessage, AIMessage(respuesta)]}
    """
    # Lista anotada: almacena todos los mensajes de la conversación
    # El reducer add_messages garantiza que los nuevos mensajes se AGREGAN
    # en lugar de reemplazar los existentes
    messages: Annotated[list, add_messages]


# =============================================================================
# SECCIÓN 6: GENERACIÓN DE EMBEDDINGS
# =============================================================================

def generate_embedding(text: str) -> List[float]:
    """
    Genera un embedding (vector numérico) para un texto usando AWS Titan.
    
    Los embeddings son representaciones vectoriales densas del significado
    semántico del texto. Textos similares producen vectores cercanos en
    el espacio vectorial, lo que permite búsquedas semánticas.
    
    CONCEPTO - EMBEDDINGS:
    ----------------------
    Un embedding transforma texto humano en un vector de números.
    Por ejemplo: "gato" -> [0.23, -0.45, 0.12, ..., 0.89] (1024 dimensiones)
    
    Textos semánticamente similares ("gato", "felino", "gatito") producen
    vectores cercanos. Esto permite encontrar documentos relevantes
    aunque no contengan las palabras exactas de la consulta.
    
    MODELO UTILIZADO - Amazon Titan Text Embeddings v2:
    ---------------------------------------------------
    - Dimensiones: 256, 512 o 1024 (usamos 1024 para mayor precisión)
    - Normalización: Los vectores se normalizan a longitud unitaria
    - Ventaja: No requiere API key adicional (usa credenciales AWS)
    
    Args:
        text (str): Texto para convertir en embedding.
                    Puede ser una pregunta del usuario o un documento.
    
    Returns:
        List[float]: Vector de 1024 dimensiones (números flotantes).
                     Este vector representa el significado semántico del texto.
    
    Ejemplo:
        embedding = generate_embedding("¿Cómo hago backup en MongoDB?")
        # Retorna: [0.023, -0.156, ..., 0.089]  # 1024 números
    """
    # PASO 1: Crear cliente de Bedrock Runtime
    # Este cliente permite invocar modelos de ML en AWS Bedrock
    # Las credenciales se obtienen automáticamente del entorno AWS
    # (variables AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.)
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",  # Servicio de inferencia de Bedrock
        region_name="us-east-1"          # Región donde está habilitado Titan
    )

    # PASO 2: Preparar el payload JSON para el modelo Titan
    # El modelo Titan v2 acepta los siguientes parámetros:
    # - inputText: El texto a convertir en embedding
    # - dimensions: Tamaño del vector (256, 512 o 1024)
    # - normalize: Si normalizar el vector a longitud 1 (recomendado)
    body = json.dumps({
        "inputText": text,      # Texto de entrada
        "dimensions": 1024,     # Máxima dimensionalidad = mayor precisión semántica
        "normalize": True       # Normalizar para que ||vector|| = 1
    })

    # PASO 3: Invocar el modelo de embeddings
    # El modelo procesa el texto y retorna el vector
    response = bedrock_runtime.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",  # ID del modelo Titan Embeddings
        contentType="application/json",           # Tipo de contenido del request
        accept="application/json",                # Tipo de respuesta esperada
        body=body                                 # Payload JSON con el texto
    )

    # PASO 4: Procesar la respuesta
    # La respuesta viene como un stream de bytes que debemos leer y parsear
    response_body = json.loads(response.get("body").read())  # Leer y parsear JSON
    embedding = response_body.get("embedding")                # Extraer el vector
    
    return embedding  # Retornar el vector de 1024 floats


# =============================================================================
# SECCIÓN 7: HERRAMIENTAS DEL AGENTE
# =============================================================================
# 
# Las herramientas (@tool) son funciones que el LLM puede decidir invocar.
# El LLM lee el docstring de cada herramienta para entender:
# 1. CUÁNDO usarla (qué tipo de preguntas puede responder)
# 2. CÓMO usarla (qué argumentos necesita)
# 3. QUÉ esperar (qué tipo de resultado retorna)
# =============================================================================

@tool
def get_information_for_question_answering(user_query: str) -> str:
    """
    Recupera documentos relevantes para responder preguntas usando búsqueda vectorial.
    
    PROPÓSITO:
    ----------
    Esta herramienta realiza una búsqueda semántica en la base de datos de
    documentación de MongoDB. Encuentra los fragmentos de texto más relevantes
    para la consulta del usuario, incluso si no contienen las palabras exactas.
    
    CUÁNDO USAR:
    ------------
    El LLM debe usar esta herramienta cuando el usuario hace preguntas que
    requieren información factual sobre MongoDB, como:
    - "¿Cuáles son las mejores prácticas para backups?"
    - "¿Cómo configuro replicación?"
    - "¿Qué es un índice compuesto?"
    
    CÓMO FUNCIONA:
    --------------
    1. Convierte la pregunta del usuario en un embedding (vector numérico)
    2. Busca en MongoDB los documentos con embeddings más similares
    3. Retorna el texto de los 5 documentos más relevantes
    
    BÚSQUEDA VECTORIAL EN MONGODB:
    ------------------------------
    MongoDB Atlas soporta índices vectoriales que permiten buscar por
    similitud semántica usando el operador $vectorSearch. El índice
    compara el vector de la consulta con los vectores almacenados.
    
    Args:
        user_query (str): La pregunta del usuario en lenguaje natural.
                          Ejemplo: "How do I configure sharding?"
    
    Returns:
        str: Texto concatenado de los 5 documentos más relevantes,
             separados por doble salto de línea.
    """
    # PASO 1: Generar embedding de la consulta
    # Convertimos la pregunta del usuario en un vector numérico
    # para poder compararlo con los embeddings de los documentos
    query_embedding = generate_embedding(user_query)

    # PASO 2: Obtener la colección de búsqueda vectorial
    # init_mongodb() retorna (cliente, vs_collection, full_collection)
    # Usamos el índice [1] para obtener la colección vectorial
    vs_collection = init_mongodb()[1]
    
    # PASO 3: Construir el pipeline de agregación
    # MongoDB usa pipelines para operaciones complejas
    # Cada elemento del pipeline es una etapa de procesamiento
    pipeline = [
        # ETAPA 1: Búsqueda Vectorial ($vectorSearch)
        {
            "$vectorSearch": {
                # Nombre del índice vectorial creado en Atlas
                "index": "vector_index",
                
                # Campo que contiene los embeddings en cada documento
                "path": "embedding",
                
                # Vector de la consulta (1024 dimensiones)
                "queryVector": query_embedding,
                
                # Número de candidatos a considerar (búsqueda más amplia)
                # Un número mayor mejora la precisión pero es más lento
                "numCandidates": 150,
                
                # Número de resultados finales a retornar
                "limit": 5,
            }
        },
        # ETAPA 2: Proyección ($project)
        # Selecciona solo los campos que necesitamos
        {
            "$project": {
                "_id": 0,    # Excluir el ID del documento (no lo necesitamos)
                "body": 1,   # Incluir el texto del documento
                # Incluir el score de similitud (0-1, mayor = más similar)
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    
    # PASO 4: Ejecutar la búsqueda
    # aggregate() ejecuta el pipeline y retorna un cursor
    results = vs_collection.aggregate(pipeline)
    
    # PASO 5: Concatenar resultados
    # Unir el texto de todos los documentos encontrados
    # El doble salto de línea ayuda al LLM a distinguir documentos
    context = "\n\n".join([doc.get("body") for doc in results])
    
    return context


@tool
def get_page_content_for_summarization(user_query: str) -> str:
    """
    Recupera el contenido completo de una página de documentación para resumir.
    
    PROPÓSITO:
    ----------
    Esta herramienta obtiene el texto completo de una página de documentación
    específica, identificada por su título exacto. Se usa cuando el usuario
    quiere un resumen de una página conocida.
    
    DIFERENCIA CON LA HERRAMIENTA QA:
    ----------------------------------
    - QA (get_information_for_question_answering): Búsqueda semántica, 
      retorna fragmentos relevantes de múltiples páginas.
    - Summarization (esta): Búsqueda exacta por título, retorna
      una página completa.
    
    CUÁNDO USAR:
    ------------
    El LLM debe usar esta herramienta cuando el usuario pide explícitamente
    un resumen de una página con título conocido:
    - "Dame un resumen de la página 'Create a MongoDB Deployment'"
    - "Summarize the 'Indexing Strategies' documentation"
    
    Args:
        user_query (str): El título EXACTO de la página de documentación.
                          Debe coincidir exactamente con el campo 'title'
                          en la base de datos.
    
    Returns:
        str: El contenido completo de la página (campo 'body'),
             o "Document not found" si no existe una página con ese título.
    """
    # PASO 1: Obtener la colección de documentos completos
    # Esta colección tiene documentos sin fragmentar, con título y cuerpo
    full_collection = init_mongodb()[2]

    # PASO 2: Construir la consulta
    # Buscamos un documento cuyo título coincida exactamente
    query = {"title": user_query}
    
    # PASO 3: Definir proyección
    # Solo necesitamos el campo 'body', excluimos '_id'
    projection = {"_id": 0, "body": 1}
    
    # PASO 4: Ejecutar búsqueda
    # find_one() retorna el primer documento que coincida, o None
    document = full_collection.find_one(query, projection)
    
    # PASO 5: Retornar resultado
    if document:
        return document["body"]  # Retornar el contenido completo
    else:
        return "Document not found"  # Mensaje de error amigable


# =============================================================================
# SECCIÓN 8: NODOS DEL GRAFO
# =============================================================================
# 
# Los nodos son funciones que procesan el estado y retornan actualizaciones.
# Cada nodo representa un paso en el flujo de decisiones del agente.
# =============================================================================

def agent(state: GraphState, llm_with_tools) -> GraphState:
    """
    Nodo del agente: invoca el LLM para decidir la siguiente acción.
    
    PROPÓSITO:
    ----------
    Este nodo es el "cerebro" del agente. Recibe el estado actual
    (historial de mensajes) y usa el LLM para:
    1. Analizar la conversación hasta el momento
    2. Decidir si necesita usar una herramienta
    3. Generar una respuesta o solicitar una herramienta
    
    FLUJO DE DECISIÓN DEL LLM:
    --------------------------
    El LLM lee todos los mensajes y decide:
    - Si falta información: genera un AIMessage con tool_calls
    - Si tiene suficiente info: genera un AIMessage con la respuesta
    
    Args:
        state (GraphState): Estado actual del grafo.
            Contiene: {"messages": [lista de mensajes]}
        llm_with_tools: Cadena LLM con herramientas vinculadas.
            Es un objeto que combina: prompt | llm.bind_tools(tools)
    
    Returns:
        GraphState: Diccionario con las actualizaciones del estado.
            Retorna: {"messages": [nuevo_mensaje_ai]}
            
            El nuevo mensaje puede ser:
            - AIMessage(content="respuesta") → ir a END
            - AIMessage(tool_calls=[...]) → ir a tools
    
    Nota sobre el reducer:
        Como messages tiene el reducer add_messages, retornar
        {"messages": [result]} AGREGA result a la lista existente,
        NO reemplaza toda la lista.
    """
    # Extraer la lista de mensajes del estado actual
    # Incluye: mensajes del usuario, respuestas previas, resultados de herramientas
    messages = state["messages"]
    
    # Invocar el LLM con el historial completo
    # El LLM analiza la conversación y genera su respuesta
    # Si decide usar una herramienta, result.tool_calls contendrá la solicitud
    result = llm_with_tools.invoke(messages)
    
    # Retornar el nuevo mensaje para agregarlo al estado
    # Nota: retornamos una lista con un solo elemento
    return {"messages": [result]}


def tool_node(state: GraphState, tools_by_name) -> GraphState:
    """
    Nodo de herramientas: ejecuta las herramientas solicitadas por el agente.
    
    PROPÓSITO:
    ----------
    Cuando el agente decide que necesita información adicional, solicita
    ejecutar una o más herramientas. Este nodo:
    1. Lee qué herramientas solicitó el agente
    2. Ejecuta cada herramienta con los argumentos proporcionados
    3. Empaqueta los resultados como ToolMessages
    
    ESTRUCTURA DE UN TOOL_CALL:
    ---------------------------
    Cada tool_call es un diccionario con:
    {
        "name": "nombre_de_la_herramienta",
        "args": {"argumento1": "valor1", ...},
        "id": "identificador_único"
    }
    
    TOOLMESSAGE:
    ------------
    Los resultados se empaquetan como ToolMessage para que el LLM
    pueda correlacionarlos con las solicitudes originales usando el ID.
    
    Args:
        state (GraphState): Estado actual del grafo.
            Contiene: {"messages": [..., AIMessage(tool_calls=[...])]}
        tools_by_name (Dict[str, Callable]): Diccionario que mapea
            nombres de herramientas a las funciones ejecutables.
            Ejemplo: {"get_information_for_question_answering": <function>}
    
    Returns:
        GraphState: Diccionario con los mensajes de resultado.
            Retorna: {"messages": [ToolMessage1, ToolMessage2, ...]}
    
    Ejemplo de flujo:
        Entrada: AIMessage con tool_calls=[{name: "get_info", args: {"query": "backup"}}]
        Proceso: ejecuta get_info("backup") -> "Los backups se configuran..."
        Salida: [ToolMessage(content="Los backups...", tool_call_id="abc123")]
    """
    # Lista para acumular los resultados de las herramientas
    result = []
    
    # Extraer las llamadas a herramientas del último mensaje (AIMessage)
    # El último mensaje es siempre el AIMessage del agente que solicitó las herramientas
    tool_calls = state["messages"][-1].tool_calls
    
    # Iterar sobre cada solicitud de herramienta
    for tool_call in tool_calls:
        # Obtener la función de herramienta por su nombre
        tool = tools_by_name[tool_call["name"]]
        
        # Ejecutar la herramienta con los argumentos proporcionados
        # invoke() llama a la función con los args como keyword arguments
        observation = tool.invoke(tool_call["args"])
        
        # Crear un ToolMessage con el resultado
        # El tool_call_id permite al LLM correlacionar solicitud y respuesta
        result.append(ToolMessage(
            content=observation,           # El resultado de la herramienta
            tool_call_id=tool_call["id"]   # ID único para correlación
        ))
    
    # Retornar todos los resultados como nuevos mensajes
    return {"messages": result}


# =============================================================================
# SECCIÓN 9: FUNCIÓN DE ENRUTAMIENTO
# =============================================================================

def route_tools(state: GraphState):
    """
    Función de enrutamiento condicional: decide el siguiente nodo.
    
    PROPÓSITO:
    ----------
    Esta función examina el último mensaje del agente y decide:
    - Si el agente solicitó herramientas → ir al nodo "tools"
    - Si el agente generó una respuesta final → ir a END
    
    CÓMO FUNCIONA EL ENRUTAMIENTO EN LANGGRAPH:
    --------------------------------------------
    add_conditional_edges() usa esta función para crear un "switch":
    - Después del nodo "agent", llama a route_tools()
    - El valor retornado ("tools" o END) determina el siguiente nodo
    - El grafo sigue el camino correspondiente
    
    Args:
        state (GraphState): Estado actual del grafo.
            Contiene: {"messages": [..., último_mensaje]}
    
    Returns:
        str: El nombre del siguiente nodo:
            - "tools": Si el AIMessage tiene tool_calls no vacíos
            - END: Si el AIMessage es una respuesta final (sin tool_calls)
    
    Raises:
        ValueError: Si no hay mensajes en el estado (caso de error)
    
    Ejemplo:
        Si último mensaje = AIMessage(content="La respuesta es...")
            → Retorna END (conversación terminada)
        Si último mensaje = AIMessage(tool_calls=[{name: "get_info", ...}])
            → Retorna "tools" (necesita ejecutar herramienta)
    """
    # Obtener la lista de mensajes, con lista vacía como default
    messages = state.get("messages", [])
    
    # Verificar que hay al menos un mensaje
    if len(messages) > 0:
        # Obtener el último mensaje (debería ser la respuesta del agente)
        ai_message = messages[-1]
    else:
        # Error: el estado debería tener mensajes en este punto
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    # Verificar si el agente solicitó herramientas
    # hasattr() verifica que el mensaje tenga el atributo tool_calls
    # También verificamos que la lista no esté vacía
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"  # Ir al nodo de herramientas
    
    # Si no hay tool_calls, el agente terminó de responder
    return END  # Ir al nodo final (terminar conversación)


# =============================================================================
# SECCIÓN 10: INICIALIZACIÓN DEL GRAFO
# =============================================================================

def init_graph(llm_with_tools, tools_by_name):
    """
    Inicializa y compila el grafo de estados del agente.
    
    PROPÓSITO:
    ----------
    Esta función ensambla todos los componentes (nodos, aristas, condiciones)
    en un grafo ejecutable. El grafo resultante puede procesar conversaciones
    de forma autónoma.
    
    ARQUITECTURA DEL GRAFO:
    -----------------------
                    ┌─────────────┐
                    │    START    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
          ┌────────│    agent    │◄──────┐
          │        └──────┬──────┘       │
          │               │              │
          │    (condicional)             │
          │     /       \\               │
          ▼    ▼         ▼              │
    ┌─────────────┐      │              │
    │    tools    │──────┘              │
    └─────────────┘                     │
                                        │
                                 ┌──────┴──────┐
                                 │     END     │
                                 └─────────────┘
    
    NODOS:
    - agent: Invoca el LLM para procesar mensajes
    - tools: Ejecuta las herramientas solicitadas
    
    ARISTAS:
    - START -> agent: Inicio de la conversación
    - agent -> tools OR END: Condicional basada en tool_calls
    - tools -> agent: Después de ejecutar herramientas
    
    Args:
        llm_with_tools: Cadena de procesamiento (prompt | llm con herramientas).
            Este objeto se pasa al nodo agent.
        tools_by_name (Dict[str, Callable]): Mapeo de nombres a funciones.
            Este diccionario se pasa al nodo tools.
    
    Returns:
        CompiledGraph: Grafo compilado listo para ejecutar.
            Se puede invocar con: app.invoke({"messages": [...]})
            O usar streaming con: app.stream({"messages": [...]})
    """
    # PASO 1: Crear instancia del grafo de estados
    # GraphState define la estructura del estado que fluye por el grafo
    graph = StateGraph(GraphState)
    
    # PASO 2: Agregar nodos
    # Cada nodo es una función que recibe state y retorna actualizaciones
    # Usamos lambdas para inyectar dependencias adicionales
    
    # Nodo del agente: usa llm_with_tools para generar respuestas
    graph.add_node("agent", lambda state: agent(state, llm_with_tools))
    
    # Nodo de herramientas: usa tools_by_name para ejecutar funciones
    graph.add_node("tools", lambda state: tool_node(state, tools_by_name))
    
    # PASO 3: Agregar aristas (conexiones entre nodos)
    
    # Arista de inicio: cuando empieza, ir al agente
    graph.add_edge(START, "agent")
    
    # Arista de regreso: después de herramientas, volver al agente
    graph.add_edge("tools", "agent")
    
    # PASO 4: Agregar aristas condicionales
    # Después del agente, usar route_tools para decidir
    # El tercer argumento mapea valores de retorno a nodos destino
    graph.add_conditional_edges(
        "agent",       # Nodo de origen
        route_tools,   # Función de enrutamiento
        {              # Mapeo de valores a nodos destino
            "tools": "tools",  # Si retorna "tools" -> ir a nodo "tools"
            END: END           # Si retorna END -> terminar grafo
        }
    )
    
    # PASO 5: Compilar el grafo
    # compile() transforma la definición en un objeto ejecutable
    return graph.compile()


# =============================================================================
# SECCIÓN 11: EJECUCIÓN DEL GRAFO
# =============================================================================

def execute_graph(app, user_input: str) -> None:
    """
    Ejecuta el grafo con una entrada del usuario y muestra los resultados.
    
    PROPÓSITO:
    ----------
    Esta función es el punto de entrada para procesar consultas.
    Toma el input del usuario, lo pasa por el grafo, y muestra
    cada paso del procesamiento incluyendo la respuesta final.
    
    STREAMING VS INVOKE:
    --------------------
    - app.invoke(): Espera hasta que termine y retorna el estado final
    - app.stream(): Retorna cada actualización de estado incrementalmente
    
    Usamos stream() para ver el progreso en tiempo real:
    - Cuándo el agente decide usar herramientas
    - Qué herramientas se ejecutan
    - Cuándo el agente genera la respuesta final
    
    FORMATO DE SALIDA:
    ------------------
    Para cada nodo que se ejecuta, imprime:
        Node agent:    (o Node tools:)
        {'messages': [...]}
    
    Al final:
        ---FINAL ANSWER---
        [contenido de la respuesta]
    
    Args:
        app: Grafo compilado (el resultado de init_graph).
        user_input (str): La consulta del usuario en lenguaje natural.
            Ejemplo: "What are best practices for MongoDB backups?"
    
    Returns:
        None: Los resultados se imprimen a consola.
    
    Nota sobre el estado de entrada:
        El input es un diccionario con la estructura del GraphState.
        El mensaje se formatea como tupla (rol, contenido) que
        LangGraph convierte automáticamente a HumanMessage.
    """
    # Preparar el estado inicial
    # La tupla ("user", texto) se convierte en HumanMessage
    input = {"messages": [("user", user_input)]}
    
    # Ejecutar el grafo en modo streaming
    # stream() retorna un generador que produce actualizaciones de estado
    for output in app.stream(input):
        # Cada output es un diccionario {nombre_nodo: actualizaciones}
        for key, value in output.items():
            print(f"Node {key}:")  # Nombre del nodo que se ejecutó
            print(value)           # Actualizaciones del estado
    
    # Imprimir separador visual
    print("---FINAL ANSWER---")
    
    # Imprimir la respuesta final
    # value es el último output (del nodo agent o del nodo final)
    # messages[-1] es el último mensaje (la respuesta del LLM)
    # .content extrae el texto de la respuesta
    print(value["messages"][-1].content)


# =============================================================================
# SECCIÓN 12: FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Función principal que orquesta la inicialización y ejecución del agente.
    
    PROPÓSITO:
    ----------
    Esta función es el punto de entrada del programa. Realiza:
    1. Inicialización de la conexión a MongoDB
    2. Configuración del prompt del sistema
    3. Vinculación de herramientas al LLM
    4. Creación del grafo de estados
    5. Ejecución de consultas de ejemplo
    
    FLUJO DE INICIALIZACIÓN:
    ------------------------
    1. Conectar a MongoDB y obtener colecciones
    2. Definir la lista de herramientas disponibles
    3. Seleccionar el LLM a usar (llm_maverick por defecto)
    4. Crear el prompt del sistema con instrucciones
    5. Vincular herramientas al LLM (bind_tools)
    6. Combinar prompt + LLM en una cadena (pipe operator)
    7. Crear diccionario de herramientas por nombre
    8. Inicializar y compilar el grafo
    9. Ejecutar consultas de ejemplo
    
    EL PROMPT DEL SISTEMA:
    ----------------------
    El prompt instruye al LLM sobre su rol y cómo comportarse:
    - Es un asistente de IA útil
    - Tiene herramientas para documentación de MongoDB
    - Debe pensar paso a paso
    - No debe re-ejecutar herramientas innecesariamente
    - Si no sabe, debe decir "I DON'T KNOW"
    
    BIND_TOOLS EXPLICADO:
    ---------------------
    bind_tools() prepara al LLM para usar herramientas:
    - Convierte las funciones @tool en esquemas JSON
    - Configura el LLM para generar tool_calls cuando necesita info
    - Mapea los docstrings para que el LLM sepa cuándo usar cada tool
    
    EL OPERADOR PIPE (|):
    ---------------------
    El operador | de LangChain crea una cadena de procesamiento:
    prompt | bind_tools significa:
    1. Primero, aplicar el prompt a los mensajes
    2. Luego, pasar el resultado al LLM
    
    Returns:
        None: Los resultados se imprimen a consola.
    """
    # PASO 1: Inicializar conexión a MongoDB
    # Obtenemos el cliente y las colecciones (aunque solo usamos el cliente aquí)
    mongodb_client, vs_collection, full_collection = init_mongodb()
    
    # PASO 2: Definir las herramientas disponibles
    # El agente podrá decidir cuál usar basándose en la consulta
    tools = [
        get_information_for_question_answering,  # Para preguntas generales
        get_page_content_for_summarization       # Para resúmenes de páginas
    ]
    
    # PASO 3: Seleccionar el modelo LLM
    # Opciones: llm_ds (DeepSeek), llm_llama (Scout), llm_maverick, llm_nova
    # llm = ChatOpenAI(openai_api_key=key_param.openai_api_key, temperature=0, model="gpt-4o")
    llm = llm_maverick  # Usando Llama 4 Maverick por defecto

    # PASO 4: Crear el prompt del sistema
    # ChatPromptTemplate permite crear prompts estructurados con variables
    prompt = ChatPromptTemplate.from_messages(
        [
            # Mensaje del sistema: define el comportamiento del asistente
            (
                "system",  # Rol del mensaje
                # Instrucciones concatenadas:
                "You are a helpful AI assistant."
                " You are provided with tools to answer questions and summarize technical documentation related to MongoDB."
                " Think step-by-step and use these tools to get the information required to answer the user query."
                " Do not re-run tools unless absolutely necessary."
                " If you are not able to get enough information using the tools, reply with I DON'T KNOW."
                " You have access to the following tools: {tool_names}."
            ),
            # Placeholder para insertar el historial de mensajes
            # variable_name debe coincidir con el campo del estado
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    # PASO 5: Insertar nombres de herramientas en el prompt
    # partial() reemplaza variables en el template
    # Resultado: "...tools: get_information_for_question_answering, get_page_content_for_summarization."
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    
    # PASO 6: Vincular herramientas al LLM
    # bind_tools() configura el LLM para usar el formato de tool_calls
    # Esto permite al LLM solicitar la ejecución de herramientas
    bind_tools = llm.bind_tools(tools)
    
    # PASO 7: Crear la cadena de procesamiento
    # El operador | conecta componentes en secuencia
    # Flujo: mensaje -> prompt -> LLM con tools -> respuesta
    llm_with_tools = prompt | bind_tools
    
    # PASO 8: Crear diccionario de herramientas
    # Permite buscar herramientas por nombre en el nodo tools
    # Ejemplo: {"get_information_for_question_answering": <function>}
    tools_by_name = {tool.name: tool for tool in tools}
    
    # PASO 9: Inicializar y compilar el grafo
    # El grafo conecta todos los componentes en un flujo de decisiones
    app = init_graph(llm_with_tools, tools_by_name)
    
    # PASO 10: Ejecutar consultas de ejemplo
    # Primera consulta: pregunta sobre mejores prácticas
    execute_graph(app, "What are some best practices for data backups in MongoDB?")
    
    # Segunda consulta: solicitud de resumen de una página
    execute_graph(app, "Give me a summary of the page titled Create a MongoDB Deployment")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
# 
# Este bloque se ejecuta cuando el archivo se ejecuta directamente
# (no cuando se importa como módulo)
# La llamada a main() inicia todo el flujo del programa
# =============================================================================

main()