# RAG with MongoDB Skill

Learn how to build a Retrieval-Augmented Generation (RAG) system using MongoDB Atlas Vector Search, LangChain, and OpenAI! This example demonstrates creating an intelligent question-answering system that can provide accurate responses based on your document content.

## What This Demo Does

📚 **Answer questions from your documents**: Ask questions about your PDF content and get intelligent responses  
🔍 **Semantic document retrieval**: Finds relevant document chunks using AI-powered similarity search  
⚡ **Fast vector search**: Uses MongoDB's optimized vector search capabilities for quick retrieval  
🤖 **AI-powered responses**: Combines retrieved context with OpenAI's GPT-4 for accurate answers  

## What You'll Need

Before getting started, make sure you have:

- ✅ **MongoDB Atlas Cluster** with connection string
- ✅ **OpenAI API Key** for GPT-4 and metadata generation
- ✅ **Voyage AI API Key** (free tier available)
- ✅ **Python 3.8+** installed on your machine

## Step-by-Step Setup

### Step 1: Set Up Your Python Environment

Create an isolated environment for this project:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 2: Install Required Packages

```bash
pip install -r requirements.txt
```

### Step 3: Configure Your API Keys

Open the `key_param.py` file and add your credentials:

```python
LLM_API_KEY="your_openai_api_key_here"
VOYAGE_API_KEY="your_voyage_api_key_here"
MONGODB_URI="your_mongodb_connection_string_here"
```

💡 **Getting your keys:**
- **MongoDB URI**: Copy from your Atlas cluster's "Connect" button
- **OpenAI API Key**: Get from [openai.com](https://platform.openai.com)
- **Voyage API Key**: Sign up at [voyageai.com](https://voyageai.com) for a free API key

## How to Use

### 1. Load Your Data

First, run the data loading script to process your PDF and store embeddings:

```bash
python load_data.py
```

⏱️ **Note**: This process may take a couple of minutes as it generates embeddings and metadata for each document chunk.

This will:
- 📄 **Load and clean** your PDF document
- ✂️ **Split text** into manageable chunks (500 chars with 150 overlap)  
- 🏷️ **Generate metadata** using OpenAI (title, keywords, hasCode)
- 🧠 **Create embeddings** using Voyage AI's voyage-3-large model
- 💾 **Store everything** in MongoDB Atlas with vector search capabilities

### 2. Create Vector Search Index

After your data is loaded, create a vector search index in your Atlas cluster's **Search & Vector Search** tab (On the left sidebar):

**Database:** `book_mongodb_chunks`  
**Collection:** `chunked_data`  
**Index Name:** `vector_index`

**Index Definition:**

```json
{
  "fields": [
    {
      "numDimensions": 1024,
      "path": "embedding",
      "similarity": "dotProduct",
      "type": "vector"
    },
    {
      "path": "hasCode",
      "type": "filter"
    }
  ]
}
```

⚠️ **Important**: Wait for the index to finish building before proceeding. You can check the index status in the Atlas UI - it should show as "Ready" before you can run queries.

### 3. Ask Questions

Run the RAG system to start asking questions:

```bash
python rag.py
```

### 4. Customize Your Queries

Edit the query in `rag.py` to ask different questions:

```python
print(query_data("What is the difference between a collection and database in MongoDB?"))
# Try other questions like:
# "How do I create an index in MongoDB?"
# "What are the benefits of using MongoDB Atlas?"
# "Explain MongoDB's aggregation pipeline"
```

### 5. View Results

The system will output intelligent answers based on your document content with:
- 💭 **Contextual answers** generated from relevant document sections
- 🎯 **Source-grounded responses** that don't hallucinate beyond your content
- ⚡ **Fast retrieval** using vector similarity search

## Example Output

```
Query: "What is the difference between a collection and database in MongoDB?"

Answer: Based on the provided context, a database in MongoDB is a container that holds collections, while a collection is a grouping of MongoDB documents. Think of a database as a filing cabinet and collections as the folders within that cabinet that organize related documents together.
```

## How It Works

1. **Document Processing**: Your PDF gets chunked into smaller pieces with metadata extraction
2. **Vector Embedding**: Each chunk gets converted to a high-dimensional vector using Voyage AI
3. **Semantic Search**: When you ask a question, it finds the most relevant chunks using vector similarity
4. **Context Assembly**: Top matching chunks get combined into context for the AI
5. **Answer Generation**: OpenAI GPT-4 generates answers based only on the retrieved context

## Troubleshooting

**🚫 "No vector index found"**: Make sure your Atlas vector search index is created and active  
**🔑 "Authentication failed"**: Verify your API keys in `key_param.py`  
**📦 "Module not found"**: Ensure you activated your virtual environment  
**📄 "File not found"**: Check that your PDF is in the `sample_files` directory

## Learn More

- 📚 [MongoDB Atlas Vector Search Documentation](https://docs.atlas.mongodb.com/atlas-vector-search/)
- 🎓 [Earn the Vector Search Fundamentals Badge](https://learn.mongodb.com/courses/vector-search-fundamentals)
- 🎓 [Earn the RAG with MongoDB Badge](https://learn.mongodb.com/courses/rag-with-mongodb)
- 🤖 [Voyage AI](https://voyageai.com/)