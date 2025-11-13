import os
import requests
import json

def get_embeddings(text, model, api_key, input_type):
    url = 'https://api.voyageai.com/v1/embeddings'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' +  api_key
    }
    data = {
        'input': text,
        'model': model,
        'input_type': input_type,
        'output_dimension': 2048
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    responseData = response.json()

    return responseData['data'][0]['embedding']
