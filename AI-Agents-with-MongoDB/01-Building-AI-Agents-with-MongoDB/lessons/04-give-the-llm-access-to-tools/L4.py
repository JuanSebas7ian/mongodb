import sys
import os

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import key_param # Importa módulo de claves.
from pymongo import MongoClient # Importa cliente MongoDB.
# from langchain.agents import tool
from langchain_core.tools import tool # Importa decorador de herramientas.
from typing import List # Importa tipado List.
# from langchain_openai import ChatOpenAI # (Comentado) Importa ChatOpenAI.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Importa componentes para prompts de chat.
# import voyageai # (Comentado) Importa VoyageAI.
import os, getpass # Importa os y getpass.
from langchain_aws import ChatBedrock # Importa cliente Bedrock.
import boto3 # Importa SDK AWS.
import json # Importa manejo de JSON.

def _set_env(var: str): # Función auxiliar set_env.
    if not os.environ.get(var): # Verifica existencia de var.
        os.environ[var] = getpass.getpass(f"{var}: ") # Solicita var si no existe.

# 1. CONFIGURACIÓN PARA DEEPSEEK-R1 (Razonamiento Complejo) # Configuración DeepSeek.
# Ideal para agentes que necesitan planificar pasos lógicos. # Descripción.
llm_ds = ChatBedrock( # Inicializa ChatBedrock DeepSeek.
    model_id="us.deepseek.r1-v1:0",  # ID oficial validado # ID modelo.
    region_name="us-east-1", # Región.
    model_kwargs={ # Argumentos.
        "temperature": 0.6, # DeepSeek recomienda 0.6 para razonamiento # Temp.
        "max_tokens": 8192,  # Recomendado para no degradar calidad del CoT # Tokens max.
        "top_p": 0.95, # Top-P.
    } # Fin kwargs.
) # Fin inicialización.


# llm = ChatBedrock( # (Comentado) Config alternativa DeepSeek V3.
#     model_id="us.deepseek.v3-v1:0", # Prueba este primero # (Comentado) ID.
#     region_name="us-east-1",        # O us-west-2 # (Comentado) Región.
#     model_kwargs={ # (Comentado) Argumentos.
#         "temperature": 0.7, # (Comentado) Temp.
#         "max_tokens": 4096 # (Comentado) Tokens.
#     } # (Comentado) Fin kwargs.
# ) # (Comentado) Fin llm.

llm_llama = ChatBedrock( # Inicializa ChatBedrock Llama 4.
    model_id="us.meta.llama4-scout-17b-instruct-v1:0",  # Nota el prefijo "us." # ID Llama 4 Scout.
    # model_id="cohere.command-r-plus-v1:0", # (Comentado) Cohere alternative.
    region_name="us-east-1", # Región AWS.
    model_kwargs={ # Argumentos.
        "temperature": 0.5, # Temperatura.
        "max_tokens": 2048, # Max tokens.
        "top_p": 0.9, # Top P.
    } # Fin kwargs.
) # Fin llm_llama.


# llm = ChatBedrock( # (Comentado) Config Llama 4 Maverick.
#     model_id="us.meta.llama4-maverick-17b-instruct-v1:0",  # Nota el prefijo "us." # (Comentado) ID.
#     region_name="us-east-1", # (Comentado) Región.
#     model_kwargs={ # (Comentado) Args.
#         "temperature": 0.5, # (Comentado) Temp.
#         "max_tokens": 2048, # (Comentado) Tokens.
#         "top_p": 0.9, # (Comentado) Top P.
#     } # (Comentado) Fin kwargs.
# ) # (Comentado) Fin.

llm_nova = ChatBedrock( # Inicializa ChatBedrock Nova Lite.
    model_id="amazon.nova-lite-v1:0",  # Nota el prefijo "us." # ID Nova Lite.
    region_name="us-east-1", # Región AWS.
    model_kwargs={ # Argumentos.
        "temperature": 0.5, # Temperatura.
        "max_tokens": 2048, # Max tokens.
        "top_p": 0.9, # Top P.
    } # Fin kwargs.
) # Fin llm_nova.

def init_mongodb(): # Función inicialización MongoDB.
    """
    Initialize MongoDB client and collections.

    Returns:
        tuple: MongoDB client, vector search collection, full documents collection.
    """ # Docstring.
    mongodb_client = MongoClient(key_param.mongodb_uri) # Cliente Mongo con URI.
    
    DB_NAME = "ai_agents" # Nombre DB.
    
    vs_collection = mongodb_client[DB_NAME]["chunked_docs"] # Colección vectores.
    
    full_collection = mongodb_client[DB_NAME]["full_docs"] # Colección documentos.
    
    return mongodb_client, vs_collection, full_collection # Retorna tupla.

    
# def generate_embedding(text: str) -> List[float]: # (Comentado) Generación embedding Voyage.
#     """
#     Generate embedding for a piece of text.

#     Args:
#         text (str): The text to embed.
#         embedding_model (voyage-3-lite): The embedding model.

#     Returns:
#         List[float]: The embedding of the text.
#     """

#     embedding_model = voyageai.Client(api_key=key_param.voyage_api_key) # (Comentado) Cliente Voyage.

#     embedding = embedding_model.embed(text, model="voyage-3-lite", input_type="query").embeddings[0] # (Comentado) Embed.
    
#     return embedding # (Comentado) Retorno.


def generate_embedding(text: str) -> List[float]: # Función embedding AWS Titan.
    """
    Generate embedding for a piece of text using AWS Titan.
    """ # Docstring.
    # 1. Crear el cliente de Bedrock Runtime # Paso 1.
    # Asegúrate de tener tus credenciales de AWS configuradas en el entorno # Recordatorio credenciales.
    bedrock_runtime = boto3.client( # Cliente boto3.
        service_name="bedrock-runtime", # Servicio.
        region_name="us-east-1"  # O tu región preferida (ej. us-west-2) # Región.
    ) # Fin cliente.

    # 2. Preparar el payload para Titan v2 # Paso 2.
    # Titan v2 soporta 'inputText' y parámetros adicionales como 'dimensions' o 'normalize' # Nota.
    body = json.dumps({ # JSON payload.
        "inputText": text, # Input texto.
        "dimensions": 1024,  # Titan v2 permite 256, 512 o 1024. 1024 es el estándar para alta calidad. # Dimensiones.
        "normalize": True # Normalizar.
    }) # Fin body.

    # 3. Invocar al modelo # Paso 3.
    response = bedrock_runtime.invoke_model( # Invocación.
        modelId="amazon.titan-embed-text-v2:0", # ID Modelo.
        contentType="application/json", # Content Type.
        accept="application/json", # Accept.
        body=body # Body.
    ) # Fin respuesta.

    # 4. Procesar la respuesta # Paso 4.
    response_body = json.loads(response.get("body").read()) # Parsea respuesta.
    embedding = response_body.get("embedding") # Obtiene embedding.
    
    return embedding # Retorna embedding.


@tool # Decorador tool.
def get_information_for_question_answering(user_query: str) -> str: # Función QA vectorial.
    """
    Retrieve relevant documents for a user query using vector search.

    Args:
        user_query (str): The user's query.

    Returns:
        str: The retrieved documents as a string.
    """ # Docstring.

    query_embedding = generate_embedding(user_query) # Genera embedding query.

    vs_collection = init_mongodb()[1] # Obtiene colección vectores.
    
    pipeline = [ # Pipeline agregación.
        { # Paso vectorSearch.
            # Use vector search to find similar documents # Comentario.
            "$vectorSearch": { # Operador.
                "index": "vector_index",  # Name of the vector index # Índice.
                "path": "embedding",       # Field containing the embeddings # Campo.
                "queryVector": query_embedding,  # The query embedding to compare against # Vector query.
                "numCandidates": 150,      # Consider 150 candidates (wider search) # Candidatos.
                "limit": 5,                # Return only top 5 matches # Límite.
            } # Fin operador.
        }, # Fin paso.
        { # Paso project.
            # Project only the fields we need # Comentario.
            "$project": { # Proyección.
                "_id": 0,                  # Exclude document ID # Sin ID.
                "body": 1,                 # Include the document body # Con body.
                "score": {"$meta": "vectorSearchScore"},  # Include the similarity score # Con score.
            } # Fin proyección.
        }, # Fin paso.
    ] # Fin pipeline.
    
    results = vs_collection.aggregate(pipeline) # Ejecuta agregación.
    
    context = "\n\n".join([doc.get("body") for doc in results]) # Une resultados.
    
    return context # Retorna contexto.

@tool  # Decorator marks this function as a tool the agent can use # Decorador tool.
def get_page_content_for_summarization(user_query: str) -> str: # Función resumen por título.
    """
    Retrieve the content of a documentation page for summarization.

    Args:
        user_query (str): The user's query (title of the documentation page).

    Returns:
        str: The content of the documentation page.
    """ # Docstring.
    full_collection = init_mongodb()[2] # Colección complete.

    query = {"title": user_query} # Query por título.
    
    projection = {"_id": 0, "body": 1} # Proyección body.
    
    document = full_collection.find_one(query, projection) # Busca uno.
    
    if document: # Si existe.
        return document["body"] # Retorna body.
    else: # Si no.
        return "Document not found" # Error.


def main(): # Función main.
    """
    Main function to initialize and execute the graph.
    """ # Docstring.
    
    mongodb_client, vs_collection, full_collection = init_mongodb() # Inicializa Mongo.
    
    tools = [ # Lista herramientas.
        get_information_for_question_answering, # Herramienta QA.
        get_page_content_for_summarization # Herramienta Resumen.
    ] # Fin lista.
    
    # llm = ChatOpenAI(openai_api_key=key_param.openai_api_key, temperature=0, model="gpt-4o") # (Comentado) LLM OpenAI.
    
    # Se debe definir qué LLM usar aquí, usaré llm_nova como default por consistencia con L2.py modificado
    llm = llm_nova # Asigna el modelo Nova Lite a la variable llm para usar en el resto de la función.

    prompt = ChatPromptTemplate.from_messages( # Crea plantilla de prompt.
        [ # Lista mensajes.
            ( # Mensaje sistema.
                "system", # Rol sistema (implícito en tupla o explícito según versión, aquí string directo asumido por LangChain).
                "You are a helpful AI assistant." # Rol asistente.
                " You are provided with tools to answer questions and summarize technical documentation related to MongoDB." # Propósito.
                " Think step-by-step and use these tools to get the information required to answer the user query." # Instrucción CoT.
                " Do not re-run tools unless absolutely necessary." # Restricción re-ejecución.
                " If you are not able to get enough information using the tools, reply with I DON'T KNOW." # Fallback.
                " You have access to the following tools: {tool_names}." # Info herramientas.
            ), # Fin mensaje sistema.
            MessagesPlaceholder(variable_name="messages"), # Placeholder historial mensajes.
        ] # Fin lista.
    ) # Fin prompt template.
    
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools])) # Rellena tool_names.
    
    bind_tools = llm.bind_tools(tools) # Vincula herramientas al LLM.
    
    llm_with_tools = prompt | bind_tools # Crea cadena prompt -> llm con herramientas.
    
    tool_call_check = llm_with_tools.invoke(["What are some best practices for data backups in MongoDB?"]).tool_calls # Invoca cadena para verificar llamada a herramientas.
    print("Tool call check:") # Imprime etiqueta.
    print(tool_call_check) # Imprime resultado de llamada.

main() # Ejecuta main.