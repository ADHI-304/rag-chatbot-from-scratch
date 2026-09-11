from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# 1. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 2. Our documents
documents = [
    "Python is a popular programming language used for artificial intelligence and machine learning.",
    "Cricket is a sport played between two teams.",
    "Java is an object-oriented programming language widely used for software development.",
    "Machine learning allows computers to learn patterns from data.",
    "Cooking involves preparing food using different ingredients and techniques."
]


# 3. User's question
query = "What programming language is commonly used for AI?"


# 4. Convert documents into embeddings
document_embeddings = model.encode(documents)


# 5. Convert question into an embedding
query_embedding = model.encode(query)


# 6. Calculate similarity between query and every document
scores = cos_sim(query_embedding, document_embeddings)[0]


# 7. Combine each document with its similarity score
results = list(zip(documents, scores))


# 8. Sort results from highest score to lowest score
results.sort(key=lambda x: x[1], reverse=True)


# 9. Select top K results
TOP_K = 3
top_results = results[:TOP_K]


# 10. Display results
print(f"Question: {query}")
print()
print(f"Top {TOP_K} most relevant documents:")
print()

for rank, (document, score) in enumerate(top_results, start=1):
    print(f"Rank {rank}")
    print(f"Score: {score:.4f}")
    print(f"Document: {document}")
    print()
