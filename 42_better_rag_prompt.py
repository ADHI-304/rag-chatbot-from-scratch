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
# 4. GET USER QUESTION
# ============================================================

question = input("Enter your question: ")


# ============================================================
# 5. RETRIEVE DOCUMENTS
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
# 8. CHECK WHETHER RELEVANT INFORMATION EXISTS
# ============================================================

if not filtered_results:

    print("\nNo sufficiently relevant information found.")
    print("I do not have enough information.")

else:

    # ========================================================
    # 9. BUILD CONTEXT
    # ========================================================

    context = "\n\n".join(
        document.page_content
        for document, score in filtered_results
    )


    # ========================================================
    # 10. BETTER RAG PROMPT
    # ========================================================

    prompt = f"""
You are a helpful and reliable Retrieval-Augmented
Generation assistant.

Your task is to answer the user's question using ONLY
the information provided in the retrieved context.

Rules:

1. Do not use outside knowledge.
2. Do not invent or guess facts.
3. Give a clear and direct answer.
4. If the context contains the answer, answer confidently.
5. If the context does not contain enough information,
   respond exactly with:
   "I do not have enough information."

Retrieved Context:
------------------
{context}
------------------

User Question:
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

    print("\nImproved RAG Answer")
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

        print(f"- {source}")
        print(f"  Distance: {score:.4f}")
