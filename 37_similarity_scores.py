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
# 5. DISPLAY RESULTS
# ============================================================

print("\nSimilarity Score Evaluation")
print("=" * 50)

for i, (document, score) in enumerate(results, start=1):

    print(f"\nResult {i}")

    print(f"Distance Score: {score:.4f}")

    print("Content:")
    print(document.page_content)

    print("-" * 50)
