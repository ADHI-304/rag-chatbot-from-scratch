
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Connect to the vector database
vectorstore = Chroma(
    persist_directory="./chroma_metadata_db",
    embedding_function=embeddings
)


# Reusable retrieval function
def retrieve_documents(question, k=3, category=None):

    search_kwargs = {
        "k": k
    }

    if category is not None:
        search_kwargs["filter"] = {
            "category": category
        }

    results = vectorstore.similarity_search(
        question,
        **search_kwargs
    )

    return results


# Get user question
question = input("Enter your question: ")


# Retrieve AI documents
results = retrieve_documents(
    question,
    k=3,
    category="AI"
)


# Display results
print("\nReusable Retriever Results")
print("=" * 50)

for i, document in enumerate(results, start=1):

    print(f"\nResult {i}")
    print("Content:", document.page_content)
    print("Metadata:", document.metadata)
    print("-" * 50)
