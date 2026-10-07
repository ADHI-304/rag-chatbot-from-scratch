from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# 1. LOAD EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 2. CONNECT TO CHROMA
# ============================================================

vectorstore = Chroma(
    persist_directory="./chroma_learning_db",
    embedding_function=embeddings
)


# ============================================================
# 3. GET USER QUESTION
# ============================================================

question = input("Enter your question: ")


# ============================================================
# 4. RETRIEVE DOCUMENTS WITH SCORES
# ============================================================

results = vectorstore.similarity_search_with_score(
    question,
    k=4
)


# ============================================================
# 5. SET EXPERIMENTAL THRESHOLD
# ============================================================

threshold = 0.8


# ============================================================
# 6. FILTER RESULTS
# ============================================================

filtered_results = [
    (document, score)
    for document, score in results
    if score <= threshold
]


# ============================================================
# 7. DISPLAY ALL RESULTS
# ============================================================

print("\nAll Retrieved Results")
print("=" * 50)

for i, (document, score) in enumerate(results, start=1):

    print(f"\nResult {i}")
    print(f"Distance: {score:.4f}")
    print(document.page_content)


# ============================================================
# 8. DISPLAY FILTERED RESULTS
# ============================================================

print("\n\nFiltered Results")
print("=" * 50)

if not filtered_results:

    print("No documents passed the relevance threshold.")

else:

    for i, (document, score) in enumerate(
        filtered_results,
        start=1
    ):

        print(f"\nResult {i}")
        print(f"Distance: {score:.4f}")
        print(document.page_content)
