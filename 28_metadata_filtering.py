
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Connect to the metadata-aware database
vectorstore = Chroma(
    persist_directory="./chroma_metadata_db",
    embedding_function=embeddings
)


# 3. Ask a question
question = input("Enter your question: ")


# 4. Retrieve only AI category documents
results = vectorstore.similarity_search(
    question,
    k=3,
    filter={"category": "AI"}
)


# 5. Display results
print("\nFiltered Retrieval Results")
print("=" * 50)

for i, document in enumerate(results, start=1):

    print(f"\nResult {i}")
    print("Content:", document.page_content)
    print("Metadata:", document.metadata)
    print("-" * 50)
