
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Connect to the existing Chroma database
vectorstore = Chroma(
    persist_directory="./chroma_learning_db",
    embedding_function=embeddings
)


# 3. Ask a question
query = "What is machine learning?"


# 4. Retrieve documents with distance scores
results = vectorstore.similarity_search_with_score(
    query,
    k=3
)


# 5. Display the results
print("Question:", query)
print()
print("Retrieved documents with scores:")

for i, (document, score) in enumerate(results, start=1):

    print(f"\n--- Result {i} ---")
    print("Distance:", round(score, 4))
    print("Content:", document.page_content)
    print("Metadata:", document.metadata)
