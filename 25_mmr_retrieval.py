
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Connect to the vector database
vectorstore = Chroma(
    persist_directory="./chroma_learning_db",
    embedding_function=embeddings
)


# 3. Get the question
question = input("Enter your question: ")


# 4. Retrieve documents using MMR
results = vectorstore.max_marginal_relevance_search(
    question,
    k=3,
    fetch_k=10,
    lambda_mult=0.5
)


# 5. Display the retrieved documents
print("\nMMR Retrieval Results")
print("=" * 50)

for i, document in enumerate(results, start=1):

    print(f"\nResult {i}")
    print("Content:")
    print(document.page_content)
    print("-" * 50)
