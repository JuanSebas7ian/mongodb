# Answer Generation Demo

This demo shows how to use RAG to answer questions from a PDF document. The demo uses a sample PDF document that contains information about MongoDB. The RAG system is used to answer questions about MongoDB from the PDF document. The demo uses the Langchain library to connect all the components of the RAG system.

## Prerequisites

- Atlas Cluster Connection String
- OpenAI API Key

## Setup

Install the requirements:

```bash
pip3 install langchain langchain_community langchain_core langchain_openai langchain_mongodb pymongo pypdf
```

Create a `key_param` file with the following content:

```python
MONGODB_URI=<your_atlas_connection_string>
LLM_API_KEY=<your_llm_api_key>
```

**Note:** Replace the `MONGODB_URI` and `LLM_API_KEY` values with your own values.

Load the sample data into your Atlas Cluster (see [Preparing The Data Demo](../L3-Preparing-The-Data/README.md)):

```bash
python load_data.py
```

Create the following vector search index on the `chunked_data` collection in your Atlas Cluster:

```bash
{
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    },
    {
      "path": "hasCode",
      "type": "filter"
    }
  ]
}
```

**Note:** Name the index `vector_index`.

## Usage

Update the `query_data` function in the `rag.py` file by passing in a question:

```python
query_data("What is the difference between a database and collection in MongoDB?")
```

Run the demo:

```bash
python rag.py
```
