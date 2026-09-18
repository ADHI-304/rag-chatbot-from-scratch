
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Connect to the vector database
vectorstore = Chroma(
    persist_directory="./chroma_learning_db",
    embedding_function=embeddings
)


# Test question and expected keyword
question = "What is machine learning?"
expected_keyword = "learn"


# Test different K values
k_values = [1, 3, 5]


print("\nHit@K Evaluation")
print("=" * 40)


for k in k_values:

    results = vectorstore.similarity_search(
        question,
        k=k
    )

    combined_content = ""

    for document in results:
        combined_content += document.page_content.lower()

    hit = expected_keyword in combined_content

    print(f"\nK = {k}")
    print("Documents retrieved:", len(results))
    print("Hit:", hit)

    if hit:
        print("Relevant keyword found.")
    else:
        print("Relevant keyword not found.")
