
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Connect to the existing vector database
vectorstore = Chroma(
    persist_directory="./chroma_learning_db",
    embedding_function=embeddings
)


# 3. Retrieve stored documents
data = vectorstore.get()


# 4. Display document information
print("\nMetadata Inspection")
print("=" * 50)

documents = data.get("documents", [])
metadatas = data.get("metadatas", [])
ids = data.get("ids", [])

print("Total documents:", len(documents))

for i in range(len(documents)):

    print(f"\nDocument {i + 1}")
    print("ID:", ids[i])
    print("Content:", documents[i])
    print("Metadata:", metadatas[i])
    print("-" * 50)
