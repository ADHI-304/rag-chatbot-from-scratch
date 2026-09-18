
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to existing Chroma database
vectorstore = Chroma(
    persist_directory="./chroma_learning_db",
    embedding_function=embeddings
)

# 3. Ask a question
query = input("Enter your question: ")

# 4. Retrieve documents with distance scores
results = vectorstore.similarity_search_with_score(
    query,
    k=4
)

# 5. Set a distance threshold
threshold = 1.0

print("\nFiltered Results:\n")

found = False

for document, distance in results:

    if distance <= threshold:

        found = True

        print("Document:")
        print(document.page_content)

        print("Distance:", round(distance, 4))
        print("-" * 40)

if not found:
    print("No sufficiently relevant documents found.")
