import sys
import os

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import key_param # Importa módulo de claves.
from pymongo import MongoClient # Cliente MongoDB.
# from langchain.agents import tool
from langchain_core.tools import tool # Decorador de herramientas.
from typing import List, Dict # Tipado List.
from typing import Annotated # Tipado Annotated.
from langgraph.graph.message import add_messages # Reducer para mensajes LangGraph.
# from langchain_openai import ChatOpenAI # (Comentado) OpenAI.
from typing_extensions import TypedDict # TypedDict para estado.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Templates de prompt.
from langchain_core.messages import ToolMessage # Mensajes de herramienta.
from langgraph.graph import END, StateGraph, START # Constantes y clases LangGraph.
# import voyageai # (Comentado) VoyageAI.
import os, getpass # Utils OS.
from langchain_aws import ChatBedrockConverse # AWS Bedrock.
import boto3 # Boto3.
import json # JSON.

def _set_env(var: str): # Función set_env.
    if not os.environ.get(var): # Verifica.
        os.environ[var] = getpass.getpass(f"{var}: ") # Solicita.

# 1. CONFIGURACIÓN PARA DEEPSEEK-R1 (Razonamiento Complejo) # Config DeepSeek.
# Ideal para agentes que necesitan planificar pasos lógicos. # Descripción.
llm_ds = ChatBedrockConverse(
    model_id="us.deepseek.r1-v1:0",  # ID oficial validado
    region_name="us-east-1",
    temperature=0.6,
    max_tokens=8192,
    top_p=0.95
)


# llm = ChatBedrockConverse( # (Comentado) Config DeepSeek V3.
#     model_id="us.deepseek.v3-v1:0", # Prueba este primero # (Comentado) ID.
#     region_name="us-east-1",        # O us-west-2 # (Comentado) Región.
#     temperature=0.7,
#     max_tokens=4096
# ) # (Comentado) Fin instancia.

llm_llama = ChatBedrockConverse(
    model_id="us.meta.llama4-scout-17b-instruct-v1:0",
    region_name="us-east-1",
    temperature=0.1,
    max_tokens=2048,
    top_p=0.9
)


llm_maverick = ChatBedrockConverse( # Config Llama 4 Maverick.
    model_id="us.meta.llama4-maverick-17b-instruct-v1:0",  # Nota el prefijo "us." # ID.
    region_name="us-east-1", # Región.
    temperature=0.5,
    max_tokens=2048,
    top_p=0.9
) # Fin instancia.

llm_nova = ChatBedrockConverse(
    model_id="amazon.nova-lite-v1:0",
    region_name="us-east-1",
    temperature=0.5,
    max_tokens=2048,
    top_p=0.9
)

def init_mongodb(): # Función inicialización Mongo.
    """
    Initialize MongoDB client and collections.

    Returns:
        tuple: MongoDB client, vector search collection, full documents collection.
    """ # Docstring.
    mongodb_client = MongoClient(key_param.mongodb_uri) # Cliente.
    
    DB_NAME = "ai_agents" # Nombre DB.
    
    vs_collection = mongodb_client[DB_NAME]["chunked_docs"] # Colección vectores.
    
    full_collection = mongodb_client[DB_NAME]["full_docs"] # Colección docs.
    
    return mongodb_client, vs_collection, full_collection # Retorna.

# Define the graph state type with messages that can accumulate # Define Estado Grafo.
class GraphState(TypedDict): # Clase TypedDict.
    # Define a messages field that keeps track of conversation history # Campo mensajes.
    messages: Annotated[list, add_messages] # Lista anotada con reducer.
    
# def generate_embedding(text: str) -> List[float]: # (Comentado) Embedding Voyage.
#     """
#     Generate embedding for a piece of text.

#     Args:
#         text (str): The text to embed.
#         embedding_model (voyage-3-lite): The embedding model.

#     Returns:
#         List[float]: The embedding of the text.
#     """

#     embedding_model = voyageai.Client(api_key=key_param.voyage_api_key) # (Comentado) Cliente.

#     embedding = embedding_model.embed(text, model="voyage-3-lite", input_type="query").embeddings[0] # (Comentado) Embed.
    
#     return embedding # (Comentado) Retorno.


def generate_embedding(text: str) -> List[float]: # Embedding Titan.
    """
    Generate embedding for a piece of text using AWS Titan.
    """ # Docstring.
    # 1. Crear el cliente de Bedrock Runtime # Paso 1.
    # Asegúrate de tener tus credenciales de AWS configuradas en el entorno # Nota.
    bedrock_runtime = boto3.client( # Cliente boto3.
        service_name="bedrock-runtime", # Servicio.
        region_name="us-east-1"  # O tu región preferida (ej. us-west-2) # Región.
    ) # Fin cliente.

    # 2. Preparar el payload para Titan v2 # Paso 2.
    # Titan v2 soporta 'inputText' y parámetros adicionales como 'dimensions' o 'normalize' # Nota.
    body = json.dumps({ # JSON body.
        "inputText": text, # Input.
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
    response_body = json.loads(response.get("body").read()) # Parsea.
    embedding = response_body.get("embedding") # Extrae.
    
    return embedding # Retorna.


@tool # Tool QA.
def get_information_for_question_answering(user_query: str) -> str: # Función.
    """
    Retrieve relevant documents for a user query using vector search.

    Args:
        user_query (str): The user's query.

    Returns:
        str: The retrieved documents as a string.
    """ # Docstring.

    query_embedding = generate_embedding(user_query) # Genera embedding.

    vs_collection = init_mongodb()[1] # Colección vectores.
    
    pipeline = [ # Pipeline.
        { # Paso vectorSearch.
            # Use vector search to find similar documents # Comentario.
            "$vectorSearch": { # Operador.
                "index": "vector_index",  # Name of the vector index # Índice.
                "path": "embedding",       # Field containing the embeddings # Campo.
                "queryVector": query_embedding,  # The query embedding to compare against # Vector.
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
    
    results = vs_collection.aggregate(pipeline) # Ejecuta.
    
    context = "\n\n".join([doc.get("body") for doc in results]) # Contexto.
    
    return context # Retorna.

@tool  # Decorator marks this function as a tool the agent can use # Tool Resumen.
def get_page_content_for_summarization(user_query: str) -> str: # Función.
    """
    Retrieve the content of a documentation page for summarization.

    Args:
        user_query (str): The user's query (title of the documentation page).

    Returns:
        str: The content of the documentation page.
    """ # Docstring.
    full_collection = init_mongodb()[2] # Colección docs.

    query = {"title": user_query} # Query.
    
    projection = {"_id": 0, "body": 1} # Proyección.
    
    document = full_collection.find_one(query, projection) # Busca.
    
    if document: # Si existe.
        return document["body"] # Retorna.
    else: # Si no.
        return "Document not found" # Error.

def agent(state: GraphState, llm_with_tools) -> GraphState: # Nodo Agente.
    """
    Agent node.

    Args:
        state (GraphState): The graph state.
        llm_with_tools: The LLM with tools.

    Returns:
        GraphState: The updated messages.
    """ # Docstring.

    messages = state["messages"] # Obtiene mensajes.
    
    result = llm_with_tools.invoke(messages) # Invoca LLM.
    
    return {"messages": [result]} # Retorna actualización.

def tool_node(state: GraphState, tools_by_name) -> GraphState: # Nodo Herramientas.
    """
    Tool node.

    Args:
        state (GraphState): The graph state.
        tools_by_name (Dict[str, Callable]): The tools by name.

    Returns:
        GraphState: The updated messages.
    """ # Docstring.
    result = [] # Lista resultados.
    
    tool_calls = state["messages"][-1].tool_calls # Obtiene llamadas a herramientas.
    
    for tool_call in tool_calls: # Itera llamadas.
        tool = tools_by_name[tool_call["name"]] # Obtiene herramienta por nombre.
        
        observation = tool.invoke(tool_call["args"]) # Invoca herramienta.
        
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"])) # Agrega mensaje herramienta con resultado.
    
    return {"messages": result} # Retorna mensajes.

def route_tools(state: GraphState): # Función enrutamiento.
    """
    Route to the tool node if the last message has tool calls. Otherwise, route to the end.

    Args:
        state (GraphState): The graph state.

    Returns:
        str: The next node to route to.
    """ # Docstring.
    messages = state.get("messages", []) # Obtiene mensajes.
    
    if len(messages) > 0: # Si hay mensajes.
        ai_message = messages[-1] # Último mensaje.
    else: # Si no.
        raise ValueError(f"No messages found in input state to tool_edge: {state}") # Error.
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0: # Si hay tool_calls.
        return "tools" # Ruta a herramientas.
    
    return END # Ruta a fin.

def init_graph(llm_with_tools, tools_by_name): # Función inicialización grafo.
    """
    Initialize the graph.

    Args:
        llm_with_tools: The LLM with tools.
        tools_by_name (Dict[str, Callable]): The tools by name.
        mongodb_client (MongoClient): The MongoDB client.

    Returns:
        StateGraph: The compiled graph.
    """ # Docstring.
    graph = StateGraph(GraphState) # Instancia grafo.
    
    graph.add_node("agent", lambda state: agent(state, llm_with_tools)) # Nodo agente.
    
    graph.add_node("tools", lambda state: tool_node(state, tools_by_name)) # Nodo herramientas.
    
    graph.add_edge(START, "agent") # Arista inicio -> agente.
    
    graph.add_edge("tools", "agent") # Arista herramientas -> agente.
    
    graph.add_conditional_edges("agent", route_tools, {"tools": "tools", END: END}) # Aristas condicionales.
    
    return graph.compile() # Compila y retorna.

def execute_graph(app, user_input: str) -> None: # Función ejecución grafo.
    """
    Stream outputs from the graph.

    Args:
        app: The compiled graph application.
        thread_id (str): The thread ID.
        user_input (str): The user's input.
    """ # Docstring.
    input = {"messages": [("user", user_input)]} # Input grafo.

    
    for output in app.stream(input): # Itera stream salida.
        for key, value in output.items(): # Itera items salida.
            print(f"Node {key}:") # Imprime nodo.
            print(value) # Imprime valor.
    
    print("---FINAL ANSWER---") # Imprime separador.
    
    print(value["messages"][-1].content) # Imprime respuesta final.

def main(): # Función main.
    """
    Main function to initialize and execute the graph.
    """ # Docstring.
    mongodb_client, vs_collection, full_collection = init_mongodb() # Inicializa Mongo.
    
    tools = [ # Lista herramientas.
        get_information_for_question_answering, # QA.
        get_page_content_for_summarization # Resumen.
    ] # Fin lista.
    
    # llm = ChatOpenAI(openai_api_key=key_param.openai_api_key, temperature=0, model="gpt-4o") # (Comentado) LLM.
    llm = llm_maverick # Asignación LLM seleccionado (usando llm_nova como default).

    prompt = ChatPromptTemplate.from_messages( # Plantilla prompt.
        [ # Mensajes.
            ( # Mensaje sistema.
                "system", # Rol.
                "You are a helpful AI assistant." # Rol.
                " You are provided with tools to answer questions and summarize technical documentation related to MongoDB." # Propósito.
                " Think step-by-step and use these tools to get the information required to answer the user query." # Instrucción.
                " Do not re-run tools unless absolutely necessary." # Restricción.
                " If you are not able to get enough information using the tools, reply with I DON'T KNOW." # Fallback.
                " You have access to the following tools: {tool_names}." # Info tools.
            ), # Fin mensaje.
            MessagesPlaceholder(variable_name="messages"), # Placeholder.
        ] # Fin lista.
    ) # Fin plantilla.
    
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools])) # Tool names.
    
    bind_tools = llm.bind_tools(tools) # Bind tools.
    
    llm_with_tools = prompt | bind_tools # Cadena.
    
    tools_by_name = {tool.name: tool for tool in tools} # Diccionario herramientas.
    
    app = init_graph(llm_with_tools, tools_by_name) # Inicializa grafo.
    
    execute_graph(app, "What are some best practices for data backups in MongoDB?") # Ejecución 1.
    
    execute_graph(app, "Give me a summary of the page titled Create a MongoDB Deployment") # Ejecución 2.
    
main() # Ejecuta main.