import sys
import os

# Add the parent directory (project root) to sys.path to allow imports from there
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import key_param
from pymongo import MongoClient
import voyageai
from datasets import load_dataset

docs = load_dataset("MongoDB/mongodb-docs")
chunked_docs = load_dataset("MongoDB/mongodb-docs-embedded")

# import voyageai
import boto3
import json

# vo = voyageai.Client(api_key=key_param.voyage_api_key)

def generate_embedding(text: str):
    """
    Generate embedding for a piece of text using AWS Titan.
    """
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )
    
    body = json.dumps({
        "inputText": text,
        "dimensions": 1024,
        "normalize": True
    })

    response = bedrock_runtime.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="application/json",
        body=body
    )

    response_body = json.loads(response.get("body").read())
    return response_body.get("embedding")

# Initialize a MongoDB Python client
mongodb_client = MongoClient(key_param.mongodb_uri)

#  Database name
DB_NAME = "ai_agents"
# Name of the collection with full documents- used for summarization
FULL_COLLECTION_NAME = "full_docs"
# Name of the collection for vector search- used for Q&A
VS_COLLECTION_NAME = "chunked_docs"
# Name of the vector search index
VS_INDEX_NAME = "vector_index"


db = mongodb_client[DB_NAME]
vs_collection = db[VS_COLLECTION_NAME]
full_collection = db[FULL_COLLECTION_NAME]

for doc in docs["train"]:
    # Insert the document into the full_docs collection
    full_collection.insert_one(doc)


for chunked_doc in chunked_docs["train"]:
    # embedding = vo.embed(chunked_doc["body"], model="voyage-3-lite", input_type="document").embeddings[0]
    embedding = generate_embedding(chunked_doc["body"])
    print(chunked_doc["body"])
    print(embedding)
    chunked_doc["embedding"] = embedding
    vs_collection.insert_one(chunked_doc)
    

model = {
    "name": VS_INDEX_NAME,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1024,
                "similarity": "cosine",
            }
        ]
    },
}

vs_collection.create_search_index(model=model) 