import requests

url = "http://localhost:11434/api/embed"

payload = {
    "model": "all-minilm",
    "input": "Vector databases store and search embeddings."
}

response = requests.post(
    url,
    json=payload,
    timeout=60
)

response.raise_for_status()

data = response.json()

embedding = data["embeddings"][0]

print("Ollama connection successful!")
print("Model: all-minilm")
print("Embedding dimensions:", len(embedding))
print("First 5 values:", embedding[:5])