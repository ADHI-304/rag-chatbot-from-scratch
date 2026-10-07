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
# 3. LOAD LLM
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
# 7. FILTER RELEVANT DOCUMENTS
# ============================================================

filtered_results = [
    (document, score)
    for document, score in results
    if score <= threshold
]


# ============================================================
# 8. CHECK RESULTS
# ============================================================

if not filtered_results:

    print("\nNo relevant documents found.")
    print("I do not have enough information.")

else:

    # ========================================================
    # 9. DISPLAY ALL RELEVANT SOURCES
    # ========================================================

    print("\nRelevant Sources")
    print("=" * 50)

    for i, (document, score) in enumerate(
        filtered_results,
        start=1
    ):

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        print(f"\nSource {i}")
        print(f"File: {source}")
        print(f"Distance: {score:.4f}")

        print("\nContent:")
        print(document.page_content)

        print("-" * 50)


    # ========================================================
    # 10. COMBINE ALL RELEVANT DOCUMENTS
    # ========================================================

    context = "\n\n".join(
        document.page_content
        for document, score in filtered_results
    )


    # ========================================================
    # 11. CREATE PROMPT
    # ========================================================

    prompt = f"""
You are a reliable RAG assistant.

Answer the user's question using ONLY the provided context.

The context may contain information from multiple documents.

Combine the relevant information when necessary.

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
    # 12. GENERATE ANSWER
    # ========================================================

    response = llm.invoke(prompt)

    answer = response.content.strip()


    # ========================================================
    # 13. DISPLAY ANSWER
    # ========================================================

    print("\nFinal Answer")
    print("=" * 50)
    print(answer)
