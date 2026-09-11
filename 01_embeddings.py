from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love programming",
    "I enjoy coding",
    "The weather is very hot today"
]

embeddings = model.encode(sentences)

similarity_1_2 = cos_sim(embeddings[0], embeddings[1])
similarity_1_3 = cos_sim(embeddings[0], embeddings[2])
similarity_2_3 = cos_sim(embeddings[1], embeddings[2])

print("Similarity between:")
print("I love programming")
print("I enjoy coding")
print(similarity_1_2)

print()

print("Similarity between:")
print("I love programming")
print("The weather is very hot today")
print(similarity_1_3)

print()

print("Similarity between:")
print("I enjoy coding")
print("The weather is very hot today")
print(similarity_2_3)
