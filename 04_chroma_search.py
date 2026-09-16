from sentence_transformers import SentenceTransformer
import chromadb


# 1. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 2. Connect to our existing Chroma database
client = chromadb.PersistentClient(
    path="./chroma_learning_db"
)


# 3. Get the existing collection
collection = client.get_collection(
    name="documents"
)


# 4. User's question
query = "What programming language is used for artificial intelligence?"


# 5. Convert the question into an embedding
query_embedding = model.encode(query).tolist()


# 6. Search Chroma
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)


# 7. Display the results
print("Question:")
print(query)
print()

print("Top 3 results:")
print()

for i, document in enumerate(results["documents"][0], start=1):
    distance = results["distances"][0][i - 1]

    print(f"Rank {i}")
    print(f"Distance: {distance:.4f}")
    print(f"Document: {document}")
    print()
