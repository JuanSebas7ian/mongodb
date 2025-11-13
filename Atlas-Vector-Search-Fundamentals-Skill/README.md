# Atlas Vector Search Fundamentals

Learn how to perform semantic search on movie plots using MongoDB Atlas Vector Search! This example demonstrates finding movies similar to your query by comparing plot descriptions using AI-generated embeddings.

## What This Demo Does

🎬 **Search movies by plot description**: Ask for "movies about escaping prison" and find relevant films  
🔍 **Semantic understanding**: Finds movies with similar themes, not just matching keywords  
⚡ **Fast vector search**: Uses MongoDB's optimized vector search capabilities  

> **📝 Note for Video Learners**  
> This code example uses pre-generated embeddings from the `sample_mflix` dataset, which differs slightly from the real-time embedding generation shown in the skill video.

## What You'll Need

Before getting started, make sure you have:

- ✅ **MongoDB Atlas Cluster** with connection string
- ✅ **Voyage AI API Key** (free tier available)
- ✅ **Python 3.7+** installed on your machine

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
pip install pymongo requests
```

### Step 3: Configure Your API Keys

Open the `key_param.py` file and add your credentials:

```python
VOYAGE_API_KEY="your_voyage_api_key_here"
MONGODB_URI="your_mongodb_connection_string_here"
```

💡 **Getting your keys:**
- **MongoDB URI**: Copy from your Atlas cluster's "Connect" button
- **Voyage API Key**: Sign up at [voyageai.com](https://voyageai.com) for a free API key

### Step 4: Load Sample Data

1. In your Atlas cluster, go to **Load Sample Dataset**
2. Load the **Sample Mflix Dataset** (contains movie data with pre-generated embeddings)

### Step 5: Create Vector Search Index

In your Atlas cluster's **Search & Vector Search** tab (On the left sidebar), create a new **Atlas Vector Search** index on the `movies` collection:

**Index Name:** `vectorPlotIndex`

**Index Definition:**
```json
{
  "fields": [
    {
      "type": "vector",
      "path": "plot_embedding_voyage_3_large",
      "numDimensions": 2048,
      "similarity": "dotProduct"
    },
    {
      "type": "filter", 
      "path": "year"
    }
  ]
}
```

## How to Use

### 1. Customize Your Search

Edit the `query` variable in `vector_search.py`:

```python
query = "A movie about people trying to escape from prison"
# Try other queries like:
# "romantic comedy in New York" 
# "space adventure with aliens"
# "detective solving a murder mystery"
```

### 2. Run the Search

```bash
python vector_search.py
```

### 3. View Results

The script will output the top 10 most similar movies with:
- 🎬 **Movie Title**
- 📝 **Plot Summary** 
- 🎯 **Similarity Score** (higher = more similar)

## Example Output

```
Title: The Shawshank Redemption
Plot: Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.
Score: 0.892

Title: Escape from Alcatraz  
Plot: A group of inmates attempt the impossible - escape from the island prison of Alcatraz.
Score: 0.847
```

## How It Works

1. **Query Processing**: Your text query gets converted to a vector embedding using Voyage AI
2. **Vector Search**: MongoDB compares your query vector against movie plot embeddings  
3. **Similarity Ranking**: Results are ranked by semantic similarity, not keyword matching
4. **Fast Results**: Vector indexes enable millisecond search across thousands of movies

## Troubleshooting

**🚫 "No results found"**: Check that your vector search index is built and active  
**🔑 "Authentication failed"**: Verify your API keys in `key_param.py`  
**📦 "Module not found"**: Make sure you activated your virtual environment  

## Learn More

- 📚 [MongoDB Atlas Vector Search Documentation](https://docs.atlas.mongodb.com/atlas-vector-search/)
- 🎓 [Earn the Vector Search Fundamentals Badge](https://learn.mongodb.com/courses/vector-search-fundamentals)
- 🤖 [Voyage AI](https://voyageai.com/)
