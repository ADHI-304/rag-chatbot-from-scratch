from sentence_transformers import SentenceTransformer
import chromadb


# 1. Our documents
documents = [
    "Python is a popular programming language used for artificial intelligence and machine learning.",
    "Cricket is a sport played between two teams.",
    "Java is an object-oriented programming language used for software development.",
    "Machine learning allows computers to learn patterns from data.",
    "Cooking involves preparing food using different ingredients and techniques."
]


# 2. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 3. Create a Chroma client
client = chromadb.PersistentClient(path="./chroma_learning_db")


# 4. Create or get a collection
collection = client.get_or_create_collection(
    name="documents"
)


# 5. Create embeddings for our documents
embeddings = model.encode(documents).tolist()


# 6. Store documents, embeddings, and IDs
collection.add(
    ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
    documents=documents,
    embeddings=embeddings
)


print("Documents stored successfully!")
print("Number of documents:", collection.count())
