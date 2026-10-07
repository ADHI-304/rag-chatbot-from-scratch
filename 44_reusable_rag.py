from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# ============================================================
# 1. CONFIGURATION
# ============================================================

THRESHOLD = 0.8


# ============================================================
# 2. LOAD EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 3. CONNECT TO CHROMA
# ============================================================

vectorstore = Chroma(
    persist_directory="./chroma_learning_db",
    embedding_function=embeddings
)


# ============================================================
# 4. LOAD LLM
# ============================================================

llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0
)


# ============================================================
# 5. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(question):

    results = vectorstore.similarity_search_with_score(
        question,
        k=4
    )

    filtered_results = [
        (document, score)
        for document, score in results
        if score <= THRESHOLD
    ]

    return filtered_results


# ============================================================
# 6. GENERATE ANSWER
# ============================================================

def generate_answer(question, documents):

    if not documents:

        return "I do not have enough information."

    context = "\n\n".join(
        document.page_content
        for document, score in documents
    )

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

    response = llm.invoke(prompt)

    return response.content.strip()


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

question = input("Enter your question: ").strip()


# ============================================================
# 8. CHECK EMPTY INPUT
# ============================================================

if not question:

    print("Error: Question cannot be empty.")

else:

    # Retrieve documents
    documents = retrieve_documents(question)

    # Generate answer
    answer = generate_answer(
        question,
        documents
    )

    # Display answer
    print("\nAnswer")
    print("=" * 50)
    print(answer)

    # Display sources
    if documents:

        print("\nSources")
        print("=" * 50)

        for document, score in documents:

            source = document.metadata.get(
                "source",
                "Unknown source"
            )

            print(f"- {source}")
            print(f"  Distance: {score:.4f}")
