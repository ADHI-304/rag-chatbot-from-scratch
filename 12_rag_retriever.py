
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


# 3. Create a reusable retrieval function
def retrieve_documents(query, k=2):

    results = vectorstore.similarity_search(
        query,
        k=k
    )

    return results


# 4. Test the retrieval function
query = "What is artificial intelligence?"

results = retrieve_documents(query, k=2)


# 5. Display retrieved documents
print("Question:", query)
print("\nRetrieved context:")

for i, document in enumerate(results, start=1):

    print(f"\n--- Document {i} ---")
    print(document.page_content)
    print("Metadata:", document.metadata)
