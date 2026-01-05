import sys
import os

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import key_param # Importa módulo de claves.
from pymongo import MongoClient # Cliente MongoDB.
# from langchain.agents import tool
from langchain_core.tools import tool # Decorador tools.
from typing import List # Tipado List.
from typing import Annotated # Tipado Annotated.
from langgraph.graph.message import add_messages # Reducer mensajes.
# from langchain_openai import ChatOpenAI # (Comentado) OpenAI.
from typing_extensions import TypedDict # TypedDict.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Prompts.
from langchain_core.messages import ToolMessage # Mensajes tool.
from langgraph.graph import END, StateGraph, START # Grafo.
from langgraph.checkpoint.mongodb import MongoDBSaver # Checkpointer MongoDB.
# import voyageai # (Comentado) Voyage.
import os, getpass # OS utils.
from langchain_aws import ChatBedrock # Bedrock.
import boto3 # Boto3.
import json # JSON.

def _set_env(var: str): # Función set env.
    if not os.environ.get(var): # Check.
        os.environ[var] = getpass.getpass(f"{var}: ") # Get pass.

# 1. CONFIGURACIÓN PARA DEEPSEEK-R1 (Razonamiento Complejo) # DeepSeek Config.
# Ideal para agentes que necesitan planificar pasos lógicos. # Desc.
llm_ds = ChatBedrock( # Instancia.
    model_id="us.deepseek.r1-v1:0",  # ID oficial validado # ID.
    region_name="us-east-1", # Región.
    model_kwargs={ # Args.
        "temperature": 0.6, # DeepSeek recomienda 0.6 para razonamiento # Temp.
        "max_tokens": 8192,  # Recomendado para no degradar calidad del CoT # Tokens.
        "top_p": 0.95, # Top P.
    } # Kwargs.
) # Fin.


# llm = ChatBedrock( # (Comentado) DeepSeek V3.
#     model_id="us.deepseek.v3-v1:0", # Prueba este primero # (Comentado) ID.
#     region_name="us-east-1",        # O us-west-2 # (Comentado) Región.
#     model_kwargs={ # (Comentado) Args.
#         "temperature": 0.7, # (Comentado) Temp.
#         "max_tokens": 4096 # (Comentado) Tokens.
#     } # (Comentado) Kwargs.
# ) # (Comentado) Fin.

llm_llama = ChatBedrock( # Instancia Llama 4.
    model_id="us.meta.llama4-scout-17b-instruct-v1:0",  # Nota el prefijo "us." # ID.
    # model_id="cohere.command-r-plus-v1:0", # (Comentado) Cohere.
    region_name="us-east-1", # Región.
    model_kwargs={ # Args.
        "temperature": 0.5, # Temp.
        "max_tokens": 2048, # Tokens.
        "top_p": 0.9, # Top P.
    } # Kwargs.
) # Fin.


# llm = ChatBedrock( # (Comentado) Llama Maverick.
#     model_id="us.meta.llama4-maverick-17b-instruct-v1:0",  # Nota el prefijo "us." # (Comentado) ID.
#     region_name="us-east-1", # (Comentado) Región.
#     model_kwargs={ # (Comentado) Args.
#         "temperature": 0.5, # (Comentado) Temp.
#         "max_tokens": 2048, # (Comentado) Tokens.
#         "top_p": 0.9, # (Comentado) Top P.
#     } # (Comentado) Kwargs.
# ) # (Comentado) Fin.

llm_nova = ChatBedrock( # Instancia Nova.
    model_id="amazon.nova-lite-v1:0",  # Nota el prefijo "us." # ID.
    region_name="us-east-1", # Región.
    model_kwargs={ # Args.
        "temperature": 0.5, # Temp.
        "max_tokens": 2048, # Tokens.
        "top_p": 0.9, # Top P.
    } # Kwargs.
) # Fin.

def init_mongodb(): # Inicializa Mongo.
    """
    Initialize MongoDB client and collections.

    Returns:
        tuple: MongoDB client, vector search collection, full documents collection.
    """ # Docstring.
    mongodb_client = MongoClient(key_param.mongodb_uri) # Cliente.
    
    DB_NAME = "ai_agents" # Nombre DB.
    
    vs_collection = mongodb_client[DB_NAME]["chunked_docs"] # Vectores.
    
    full_collection = mongodb_client[DB_NAME]["full_docs"] # Docs.
    
    return mongodb_client, vs_collection, full_collection # Retorno.

# Define the graph state type with messages that can accumulate # Estado Grafo.
class GraphState(TypedDict): # Clase TypedDict.
    # Define a messages field that keeps track of conversation history # Campo mensajes.
    messages: Annotated[list, add_messages] # Lista anotada.
    
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
    ) # Fin invocación.

    # 4. Procesar la respuesta # Paso 4.
    response_body = json.loads(response.get("body").read()) # Parsea.
    embedding = response_body.get("embedding") # Extrae.
    
    return embedding # Retorna.


@tool # Tool QA.
def get_information_for_question_answering(user_query: str) -> str: # Función QA.
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
def get_page_content_for_summarization(user_query: str) -> str: # Función Resumen.
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

    messages = state["messages"] # Mensajes.
    
    result = llm_with_tools.invoke(messages) # Invoca LLM.
    
    return {"messages": [result]} # Retorna.

def tool_node(state: GraphState, tools_by_name) -> GraphState: # Nodo Herramientas.
    """
    Tool node.

    Args:
        state (GraphState): The graph state.
        tools_by_name (Dict[str, Callable]): The tools by name.

    Returns:
        GraphState: The updated messages.
    """ # Docstring.
    result = [] # Lista.
    
    tool_calls = state["messages"][-1].tool_calls # Tool calls.
    
    for tool_call in tool_calls: # Itera.
        tool = tools_by_name[tool_call["name"]] # Tool by name.
        
        observation = tool.invoke(tool_call["args"]) # Invoca.
        
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"])) # Agrega observación.
    
    return {"messages": result} # Retorna.

def route_tools(state: GraphState): # Enrutamiento.
    """
    Route to the tool node if the last message has tool calls. Otherwise, route to the end.

    Args:
        state (GraphState): The graph state.

    Returns:
        str: The next node to route to.
    """ # Docstring.
    messages = state.get("messages", []) # Mensajes.
    
    if len(messages) > 0: # Si hay.
        ai_message = messages[-1] # Último.
    else: # Si no.
        raise ValueError(f"No messages found in input state to tool_edge: {state}") # Error.
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0: # Check tool calls.
        return "tools" # Ruta tools.
    
    return END # Ruta fin.

def init_graph(llm_with_tools, tools_by_name, mongodb_client): # Init grafo con persistencia.
    """
    Initialize the graph.

    Args:
        llm_with_tools: The LLM with tools.
        tools_by_name (Dict[str, Callable]): The tools by name.
        mongodb_client (MongoClient): The MongoDB client.

    Returns:
        StateGraph: The compiled graph.
    """ # Docstring.
    graph = StateGraph(GraphState) # Grafo.
    
    graph.add_node("agent", lambda state: agent(state, llm_with_tools)) # Nodo agente.
    
    graph.add_node("tools", lambda state: tool_node(state, tools_by_name)) # Nodo tools.
    
    graph.add_edge(START, "agent") # Inicio -> Agente.
    
    graph.add_edge("tools", "agent") # Tools -> Agente.
    
    graph.add_conditional_edges("agent", route_tools, {"tools": "tools", END: END}) # Condicional.
    
    checkpointer = MongoDBSaver(mongodb_client) # Checkpointer Mongo.
    
    return graph.compile(checkpointer=checkpointer) # Compila con checkpointer.

def execute_graph(app, thread_id: str, user_input: str) -> None: # Ejecución grafo.
    """
    Stream outputs from the graph.

    Args:
        app: The compiled graph application.
        thread_id (str): The thread ID.
        user_input (str): The user's input.
    """ # Docstring.
    input = {"messages": [("user", user_input)]} # Input.
    
    config = {"configurable": {"thread_id": thread_id}} # Config thread.
    
    for output in app.stream(input, config): # Stream.
        for key, value in output.items(): # Items.
            print(f"Node {key}:") # Imprime nodo.
            print(value) # Imprime valor.
    
    print("---FINAL ANSWER---") # Final.
    
    print(value["messages"][-1].content) # Contenido final.

def main(): # Main.
    """
    Main function to initialize and execute the graph.
    """ # Docstring.
    mongodb_client, vs_collection, full_collection = init_mongodb() # Init Mongo.
    
    tools = [ # Tools.
        get_information_for_question_answering, # QA.
        get_page_content_for_summarization # Resumen.
    ] # Fin lista.
    
    # llm = ChatOpenAI(openai_api_key=key_param.openai_api_key, temperature=0, model="gpt-4o") # (Comentado) LLM.
    llm = llm_nova # LLM seleccionado.

    prompt = ChatPromptTemplate.from_messages( # Prompt.
        [ # Mensajes.
            ( # Sistema.
                "system", # Rol.
                "You are a helpful AI assistant." # Rol.
                " You are provided with tools to answer questions and summarize technical documentation related to MongoDB." # Propósito.
                " Think step-by-step and use these tools to get the information required to answer the user query." # Instrucción.
                " Do not re-run tools unless absolutely necessary." # Restricción.
                " If you are not able to get enough information using the tools, reply with I DON'T KNOW." # Fallback.
                " You have access to the following tools: {tool_names}." # Info.
            ), # Fin sistema.
            MessagesPlaceholder(variable_name="messages"), # Placeholder.
        ] # Fin lista.
    ) # Fin prompt.
    
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools])) # Tool names.
    
    bind_tools = llm.bind_tools(tools) # Bind.
    
    llm_with_tools = prompt | bind_tools # Pipe.
    
    tools_by_name = {tool.name: tool for tool in tools} # Map tools.
    
    app = init_graph(llm_with_tools, tools_by_name, mongodb_client) # Init grafo.
    
    execute_graph(app, "1", "What are some best practices for data backups in MongoDB?") # Run 1.
    execute_graph(app, "1", "What did I just ask you?") # Run 2.
main() # Run main.