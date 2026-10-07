from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


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
# 3. LOAD OLLAMA
# ============================================================

llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0
)


# ============================================================
# 4. GET QUESTION
# ============================================================

question = input("Enter your question: ")


# ============================================================
# 5. RETRIEVE DOCUMENTS WITH SCORES
# ============================================================

results = vectorstore.similarity_search_with_score(
    question,
    k=4
)


# ============================================================
# 6. RELEVANCE THRESHOLD
# ============================================================

threshold = 0.8


# ============================================================
# 7. FILTER RESULTS
# ============================================================

filtered_results = [
    (document, score)
    for document, score in results
    if score <= threshold
]


# ============================================================
# 8. HANDLE NO RELEVANT RESULTS
# ============================================================

if not filtered_results:

    print("\nNo sufficiently relevant information found.")
    print("I do not have enough information.")

else:

    # ========================================================
    # 9. CREATE CONTEXT
    # ========================================================

    context = "\n\n".join(
        document.page_content
        for document, score in filtered_results
    )


    # ========================================================
    # 10. CREATE PROMPT
    # ========================================================

    prompt = f"""
You are a reliable RAG assistant.

Answer the question using ONLY the provided context.

Do not invent facts.

If the context does not contain enough information,
say:

"I do not have enough information."

Context:
{context}

Question:
{question}

Answer:
"""


    # ========================================================
    # 11. GENERATE ANSWER
    # ========================================================

    response = llm.invoke(prompt)

    answer = response.content.strip()


    # ========================================================
    # 12. DISPLAY ANSWER
    # ========================================================

    print("\nAnswer")
    print("=" * 50)
    print(answer)


    # ========================================================
    # 13. DISPLAY SOURCES
    # ========================================================

    print("\nSources")
    print("=" * 50)

    for document, score in filtered_results:

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        print(f"\nSource: {source}")
        print(f"Distance: {score:.4f}")

        print("\nRetrieved Content:")
        print(document.page_content)

        print("-" * 50)
