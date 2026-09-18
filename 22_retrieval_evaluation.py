
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Connect to Chroma
vectorstore = Chroma(
    persist_directory="./chroma_learning_db",
    embedding_function=embeddings
)


# 3. Get user question
query = input("Enter your question: ")


# 4. Retrieve documents with scores
results = vectorstore.similarity_search_with_score(
    query,
    k=4
)


# 5. Display every retrieved document
print("\nRetrieval Evaluation\n")

for i, (document, distance) in enumerate(results, start=1):

    print(f"Result {i}")
    print(f"Distance: {distance:.4f}")
    print("Content:")
    print(document.page_content)
    print("-" * 50)
