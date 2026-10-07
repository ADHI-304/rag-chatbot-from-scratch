from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# ============================================================
# 1. LOAD EMBEDDINGS
# ============================================================

try:

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

except Exception as e:

    print("Error loading embedding model.")
    print(f"Details: {e}")
    exit()


# ============================================================
# 2. CONNECT TO CHROMA
# ============================================================

try:

    vectorstore = Chroma(
        persist_directory="./chroma_learning_db",
        embedding_function=embeddings
    )

except Exception as e:

    print("Error connecting to Chroma database.")
    print(f"Details: {e}")
    exit()


# ============================================================
# 3. LOAD LLM
# ============================================================

try:

    llm = ChatOllama(
        model="qwen2.5-coder:7b",
        temperature=0
    )

except Exception as e:

    print("Error loading LLM.")
    print(f"Details: {e}")
    exit()


# ============================================================
# 4. GET USER QUESTION
# ============================================================

question = input("Enter your question: ").strip()


# ============================================================
# 5. CHECK EMPTY INPUT
# ============================================================

if not question:

    print("\nError: Question cannot be empty.")
    exit()


# ============================================================
# 6. RETRIEVE DOCUMENTS
# ============================================================

try:

    results = vectorstore.similarity_search_with_score(
        question,
        k=4
    )

except Exception as e:

    print("\nError during document retrieval.")
    print(f"Details: {e}")
    exit()


# ============================================================
# 7. FILTER RESULTS
# ============================================================

threshold = 0.8

filtered_results = [
    (document, score)
    for document, score in results
    if score <= threshold
]


# ============================================================
# 8. CHECK RELEVANT RESULTS
# ============================================================

if not filtered_results:

    print("\nNo sufficiently relevant information found.")
    print("I do not have enough information.")
    exit()


# ============================================================
# 9. BUILD CONTEXT
# ============================================================

context = "\n\n".join(
    document.page_content
    for document, score in filtered_results
)


# ============================================================
# 10. CREATE PROMPT
# ============================================================

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


# ============================================================
# 11. GENERATE ANSWER
# ============================================================

try:

    response = llm.invoke(prompt)

    answer = response.content.strip()

except Exception as e:

    print("\nError generating answer.")
    print(f"Details: {e}")
    exit()


# ============================================================
# 12. DISPLAY ANSWER
# ============================================================

print("\nAnswer")
print("=" * 50)
print(answer)


# ============================================================
# 13. DISPLAY SOURCES
# ============================================================

print("\nSources")
print("=" * 50)

for document, score in filtered_results:

    source = document.metadata.get(
        "source",
        "Unknown source"
    )

    print(f"- {source}")
    print(f"  Distance: {score:.4f}")
