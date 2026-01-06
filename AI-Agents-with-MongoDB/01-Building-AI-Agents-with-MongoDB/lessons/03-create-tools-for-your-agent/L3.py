import sys
import os

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import key_param # Importa el módulo key_param que contiene las claves y configuraciones.
from pymongo import MongoClient # Importa MongoClient para conectar con MongoDB.
# from langchain.agents import tool # Deprecated import
from langchain_core.tools import tool # Importa el decorador @tool para convertir funciones en herramientas.
from typing import List # Importa List para el tipado de listas en Python.
# from langchain_openai import ChatOpenAI # (Comentado) Importa ChatOpenAI de langchain_openai.
# import voyageai # (Comentado) Importa la librería voyageai para embeddings.
import os, getpass # Importa os para sistema y getpass para contraseñas seguras.
from langchain_aws import ChatBedrockConverse # Importa ChatBedrockConverse para usar modelos de AWS Bedrock.

def _set_env(var: str): # Función para configurar variables de entorno si no existen.
    if not os.environ.get(var): # Comprueba si la variable de entorno 'var' no existe.
        os.environ[var] = getpass.getpass(f"{var}: ") # Pide la variable y la guarda en el entorno.

# 1. CONFIGURACIÓN PARA DEEPSEEK-R1 (Razonamiento Complejo) # Configuración del modelo DeepSeek-R1.
# Ideal para agentes que necesitan planificar pasos lógicos. # Descripción del uso del modelo.
llm_ds = ChatBedrockConverse(
    model_id="us.deepseek.r1-v1:0",  # ID oficial validado
    region_name="us-east-1",
    temperature=0.6,
    max_tokens=8192,
    top_p=0.95
)


# llm = ChatBedrockConverse( # (Comentado) Configuración alternativa para DeepSeek V3.
#     model_id="us.deepseek.v3-v1:0", # Prueba este primero # (Comentado) ID del modelo.
#     region_name="us-east-1",        # O us-west-2 # (Comentado) Región.
#     temperature=0.7,
#     max_tokens=4096
# ) # (Comentado) Fin llm.

llm_llama = ChatBedrockConverse(
    model_id="us.meta.llama4-scout-17b-instruct-v1:0",
    region_name="us-east-1",
    temperature=0.1,
    max_tokens=2048,
    top_p=0.9
)


# llm = ChatBedrockConverse( # (Comentado) Configuración alternativa para Llama 4 Maverick.
#     model_id="us.meta.llama4-maverick-17b-instruct-v1:0",  # Nota el prefijo "us." # (Comentado) ID del modelo.
#     region_name="us-east-1", # (Comentado) Región.
#     temperature=0.5,
#     max_tokens=2048,
#     top_p=0.9
# ) # (Comentado) Fin llm.

llm_nova = ChatBedrockConverse(
    model_id="amazon.nova-lite-v1:0",
    region_name="us-east-1",
    temperature=0.5,
    max_tokens=2048,
    top_p=0.9
)

import json # Importa el módulo json para manejar datos JSON.
import boto3 # Importa la librería boto3 para interactuar con servicios de AWS.
from typing import List # Importa List (redundante, ya importado arriba).

def init_mongodb(): # Define función para inicializar MongoDB.
    """
    Initialize MongoDB client and collections.

    Returns:
        tuple: MongoDB client, vector search collection, full documents collection.
    """ # Docstring de la función.
    mongodb_client = MongoClient(key_param.mongodb_uri) # Crea cliente MongoDB con URI.
    
    DB_NAME = "ai_agents" # Define nombre de la base de datos.
    
    vs_collection = mongodb_client[DB_NAME]["chunked_docs"] # Obtiene colección para búsqueda vectorial.
    
    full_collection = mongodb_client[DB_NAME]["full_docs"] # Obtiene colección de documentos completos.
    
    return mongodb_client, vs_collection, full_collection # Retorna cliente y colecciones.

# def generate_embedding(text: str) -> List[float]: # (Comentado) Función para generar embeddings con Voyage AI.
#     """
#     Generate embedding for a piece of text.

#     Args:
#         text (str): The text to embed.
#         embedding_model (voyage-3-lite): The embedding model.

#     Returns:
#         List[float]: The embedding of the text.
#     """

#     embedding_model = voyageai.Client(api_key=key_param.voyage_api_key) # (Comentado) Cliente VoyageAI.

#     embedding = embedding_model.embed(text, model="voyage-3-lite", input_type="query").embeddings[0] # (Comentado) Generación embedding.
    
#     return embedding # (Comentado) Retorno.


def generate_embedding(text: str) -> List[float]: # Define función para generar embeddings con AWS Titan.
    """
    Generate embedding for a piece of text using AWS Titan.
    """ # Docstring explicativo.
    # 1. Crear el cliente de Bedrock Runtime # Paso 1: Cliente Bedrock.
    # Asegúrate de tener tus credenciales de AWS configuradas en el entorno # Nota sobre credenciales.
    bedrock_runtime = boto3.client( # Crea cliente boto3 para bedrock-runtime.
        service_name="bedrock-runtime", # Nombre del servicio.
        region_name="us-east-1"  # O tu región preferida (ej. us-west-2) # Región AWS.
    ) # Fin creación cliente.

    # 2. Preparar el payload para Titan v2 # Paso 2: Payload JSON.
    # Titan v2 soporta 'inputText' y parámetros adicionales como 'dimensions' o 'normalize' # Comentario.
    body = json.dumps({ # Serializa diccionario a JSON string.
        "inputText": text, # Texto a embeder.
        "dimensions": 1024,  # Titan v2 permite 256, 512 o 1024. 1024 es el estándar para alta calidad. # Dimensión del vector.
        "normalize": True # Normalizar vector resultante.
    }) # Fin payload.

    # 3. Invocar al modelo # Paso 3: Invocación.
    response = bedrock_runtime.invoke_model( # Llama al modelo en Bedrock.
        modelId="amazon.titan-embed-text-v2:0", # ID del modelo de embeddings Titan V2.
        contentType="application/json", # Tipo de contenido de la solicitud.
        accept="application/json", # Tipo de contenido aceptado en respuesta.
        body=body # Cuerpo de la solicitud.
    ) # Fin invocación.

    # 4. Procesar la respuesta # Paso 4: Procesamiento.
    response_body = json.loads(response.get("body").read()) # Lee y parsea el cuerpo de la respuesta JSON.
    embedding = response_body.get("embedding") # Extrae el campo 'embedding'.
    
    return embedding # Retorna el embedding (lista de floats).

@tool # Decorador para marcar la función como una herramienta para el agente LangChain.
def get_information_for_question_answering(user_query: str) -> str: # Define herramienta de búsqueda vectorial.
    """
    Retrieve relevant documents for a user query using vector search.

    Args:
        user_query (str): The user's query.

    Returns:
        str: The retrieved documents as a string.
    """ # Docstring de la herramienta.

    query_embedding = generate_embedding(user_query) # Genera embedding para la consulta del usuario.

    vs_collection = init_mongodb()[1] # Obtiene la colección de búsqueda vectorial desde init_mongodb.
    
    pipeline = [ # Define el pipeline de agregación para MongoDB.
        { # Etapa 1: Búsqueda vectorial.
            # Use vector search to find similar documents # Comentario.
            "$vectorSearch": { # Operador de búsqueda vectorial.
                "index": "vector_index",  # Name of the vector index # Nombre del índice vectorial.
                "path": "embedding",       # Field containing the embeddings # Campo en el documento.
                "queryVector": query_embedding,  # The query embedding to compare against # Vector de consulta.
                "numCandidates": 150,      # Consider 150 candidates (wider search) # Candidatos a considerar.
                "limit": 5,                # Return only top 5 matches # Límite de resultados.
            } # Fin operador.
        }, # Fin etapa.
        { # Etapa 2: Proyección.
            # Project only the fields we need # Comentario.
            "$project": { # Operador de proyección.
                "_id": 0,                  # Exclude document ID # Excluye ID.
                "body": 1,                 # Include the document body # Incluye cuerpo.
                "score": {"$meta": "vectorSearchScore"},  # Include the similarity score # Incluye score de similitud.
            } # Fin operador.
        }, # Fin etapa.
    ] # Fin pipeline.
    
    results = vs_collection.aggregate(pipeline) # Ejecuta la agregación en la colección.
    
    context = "\n\n".join([doc.get("body") for doc in results]) # Concatena los cuerpos de los documentos encontrados.
    
    return context # Retorna el contexto como string único.

@tool # Decorador para herramienta.
def get_page_content_for_summarization(user_query: str) -> str: # Define herramienta para obtener contenido por título.
    """
    Retrieve the content of a documentation page for summarization.

    Args:
        user_query (str): The user's query (title of the documentation page).

    Returns:
        str: The content of the documentation page.
    """ # Docstring explicativo.
    full_collection = init_mongodb()[2] # Obtiene colección de documentos completos.

    query = {"title": user_query} # Construye consulta para buscar por título exacto.
    
    projection = {"_id": 0, "body": 1} # Define proyección para obtener solo el cuerpo.
    
    document = full_collection.find_one(query, projection) # Busca un documento que coincida.
    
    if document: # Si se encuentra el documento.
        return document["body"] # Retorna su cuerpo.
    else: # Si no se encuentra.
        return "Document not found" # Retorna mensaje de error.

def main(): # Función principal para probar las herramientas.
    """
    Main function to initialize and execute the graph.
    """ # Docstring.
    # Initialize MongoDB connections # Comentario.
    mongodb_client, vs_collection, full_collection = init_mongodb() # Inicializa conexiones.
    
    # Initialize the ChatOpenAI model with API key # Comentario original.
    # Initialize the ChatOpenAI model with API key # Comentario duplicado.
    # llm = ChatOpenAI(openai_api_key=key_param.openai_api_key, temperature=0, model="gpt-4o") # (Comentado) Inicialización OpenAI.
    llm = ChatBedrock( # Inicialización de ChatBedrock para Llama 4.
            model_id="us.meta.llama4-scout-17b-instruct-v1:0",  # Nota el prefijo "us." # ID del modelo Llama 4 Scout.
            region_name="us-east-1", # Región de AWS.
            model_kwargs={ # Argumentos adicionales.
                "temperature": 0.5, # Temperatura para el modelo.
                "max_tokens": 2048, # Tokens máximos de respuesta.
                "top_p": 0.9, # Parámetro Top-P.
            } # Fin kwargs.
        ) # Fin inicialización llm.

    tools = [ # Lista de herramientas disponibles.
        get_information_for_question_answering, # Herramienta de QA.
        get_page_content_for_summarization # Herramienta de resumen.
    ] # Fin lista tools.

    answer = get_information_for_question_answering.invoke( # Invoca la herramienta de QA manualmente.
    "What are some best practices for data backups in MongoDB?" # Argumento de consulta.
    ) # Fin invocación.
    print("answer:" + answer) # Imprime la respuesta obtenida.

    summary = get_page_content_for_summarization.invoke("Create a MongoDB Deployment") # Invoca la herramienta de resumen manualmente.
    print("Summary:" + summary) # Imprime el resumen obtenido.
    

# Execute main function when script is run directly # Comentario.
main() # Ejecuta main.