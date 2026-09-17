
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


# 4. Retrieve the most relevant documents
results = vectorstore.similarity_search(
    query,
    k=2
)


# 5. Display the results
print("Question:", query)
print()
print("Retrieved documents:")

for i, document in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print("Content:", document.page_content)
    print("Metadata:", document.metadata)
