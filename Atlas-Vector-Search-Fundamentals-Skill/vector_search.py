from pymongo import MongoClient
from embeddings import get_embeddings

import key_param

client = MongoClient(key_param.MONGODB_URI)
db_name = "sample_mflix"
collection_name = "embedded_movies"
model = "voyage-3-large"
collection = client[db_name][collection_name]

query = "A movie about people who are trying to escape from a maximum security facility."
input_type = "query"
embedding = get_embeddings(query, model, key_param.VOYAGE_API_KEY, input_type)

pipeline = [
    {
        '$vectorSearch': {
            'exact': False, # Set to True to use ENN
            'index': 'vectorPlotIndex',
            'path': 'plot_embedding_voyage_3_large',
            'queryVector': embedding,
            'numCandidates': 200,
            'limit': 10,
            # 'filter': {
            #     'year': {
            #         '$gt’: 2010'
            #     }
            # }
        }
    },
    {
      '$project': {
          'title': 1,
          'plot': 1,
          'score': {
              '$meta': 'vectorSearchScore'
          }
      }
    }
]

results = collection.aggregate(pipeline)
for doc in results:
    print(f"Title: {doc['title']}")
    print(f"Plot: {doc['plot']}")
    print(f"Score: {doc['score']}")
    
