import sys
import os

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import key_param # Importa el módulo key_param que contiene claves y URIs de configuración.
from pymongo import MongoClient # Importa la clase MongoClient de la librería pymongo para interactuar con bases de datos MongoDB.
# from langchain_openai import ChatOpenAI # (Comentado) Importa ChatOpenAI de langchain_openai, usado anteriormente para modelos de OpenAI.
from langchain_aws import ChatBedrockConverse
import os, getpass # Importa los módulos os (interacción con sistema operativo) y getpass (entrada segura de contraseñas).

def _set_env(var: str): # Define una función auxiliar para configurar variables de entorno si no existen.
    if not os.environ.get(var): # Verifica si la variable de entorno 'var' no está definida actualmente.
        os.environ[var] = getpass.getpass(f"{var}: ") # Solicita al usuario el valor de 'var' de forma segura y lo asigna a las variables de entorno.

# 1. CONFIGURACIÓN PARA DEEPSEEK-R1 (Razonamiento Complejo) # Comentario explicativo sobre la configuración del modelo DeepSeek-R1.
# Ideal para agentes que necesitan planificar pasos lógicos. # Comentario sobre el caso de uso ideal para este modelo.
llm_ds = ChatBedrockConverse(
    model_id="us.deepseek.r1-v1:0",  # ID oficial validado
    region_name="us-east-1",
    temperature=0.6,
    max_tokens=8192,
    top_p=0.95
)


# llm = ChatBedrockConverse( # (Comentado) Inicialización alternativa para DeepSeek V3.
#     model_id="us.deepseek.v3-v1:0", # Prueba este primero # (Comentado) ID del modelo DeepSeek V3.
#     region_name="us-east-1",        # O us-west-2 # (Comentado) Región de AWS.
#     temperature=0.7,
#     max_tokens=4096
# ) # (Comentado) Cierre de inicialización.

llm_llama = ChatBedrockConverse(
    model_id="us.meta.llama4-scout-17b-instruct-v1:0",
    region_name="us-east-1",
    temperature=0.1,
    max_tokens=2048,
    top_p=0.9
)


# llm = ChatBedrockConverse( # (Comentado) Inicialización alternativa para Llama 4 Maverick.
#     model_id="us.meta.llama4-maverick-17b-instruct-v1:0",  # Nota el prefijo "us." # (Comentado) ID del modelo Maverick.
#     region_name="us-east-1", # (Comentado) Región.
#     temperature=0.5,
#     max_tokens=2048,
#     top_p=0.9
# ) # (Comentado) Cierre de inicialización.

llm_nova = ChatBedrockConverse(
    model_id="amazon.nova-lite-v1:0",
    region_name="us-east-1",
    temperature=0.5,
    max_tokens=2048,
    top_p=0.9
)

def init_mongodb(): # Define la función init_mongodb para configurar la conexión a la base de datos.
    """
    Initialize MongoDB client and collections.
    
    Returns:
        tuple: MongoDB client, vector search collection, full documents collection.
    """ # Docstring que explica el propósito y el retorno de la función.
    mongodb_client = MongoClient(key_param.mongodb_uri) # Crea un cliente de MongoDB usando la URI almacenada en key_param.
    
    DB_NAME = "ai_agents" # Define el nombre de la base de datos a usar ("ai_agents").
    
    vs_collection = mongodb_client[DB_NAME]["chunked_docs"] # Accede a la colección "chunked_docs" dentro de la base de datos para búsqueda vectorial.
    
    full_collection = mongodb_client[DB_NAME]["full_docs"] # Accede a la colección "full_docs" dentro de la base de datos para documentos completos.
    
    return mongodb_client, vs_collection, full_collection # Retorna el cliente y las dos colecciones.

def main(): # Define la función principal del script.
    """
    Main function to initialize and execute the graph.
    """ # Docstring de la función main.
    # Initialize MongoDB connections # Comentario: Inicializar conexiones a MongoDB.
    mongodb_client, vs_collection, full_collection = init_mongodb() # Llama a init_mongodb y desempaqueta los retornos.
    
    # Initialize the ChatOpenAI model with API key # Comentario original sobre ChatOpenAI (mantenido).
    # Initialize the ChatOpenAI model with API key # Comentario duplicado (mantenido).
    # llm = ChatOpenAI(openai_api_key=key_param.openai_api_key, temperature=0, model="gpt-4o") # (Comentado) Inicialización original de ChatOpenAI con clave y modelo GPT-4o.
    llm = ChatBedrockConverse(
        model_id="amazon.nova-lite-v1:0",
        region_name="us-east-1",
        temperature=0.5,
        max_tokens=2048,
        top_p=0.9
)
    

# Execute main function when script is run directly # Comentario: Ejecutar main si el script se corre directamente.
main() # Llama a la función main para iniciar el programa.