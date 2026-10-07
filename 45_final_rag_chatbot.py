from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# ============================================================
# CONFIGURATION
# ============================================================

THRESHOLD = 0.8
TOP_K = 4


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# CONNECT TO VECTOR DATABASE
# ============================================================

vectorstore = Chroma(
    persist_directory="./chroma_learning_db",
    embedding_function=embeddings
)


# ============================================================
# LOAD LLM
# ============================================================

llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0
)


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(question):

    results = vectorstore.similarity_search_with_score(
        question,
        k=TOP_K
    )

    filtered_results = [
        (document, score)
        for document, score in results
        if score <= THRESHOLD
    ]

    return filtered_results


# ============================================================
# GENERATE RAG ANSWER
# ============================================================

def generate_answer(question, documents):

    if not documents:

        return "I do not have enough information."

    context = "\n\n".join(
        document.page_content
        for document, score in documents
    )

    prompt = f"""
You are a reliable Retrieval-Augmented Generation assistant.

Answer the user's question using ONLY the retrieved context.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Give a clear and direct answer.
4. If the context does not contain enough information,
   say exactly:
   "I do not have enough information."

Retrieved Context:
------------------
{context}
------------------

User Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content.strip()


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(documents):

    if not documents:
        return

    print("\nSources")
    print("=" * 50)

    for document, score in documents:

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        print(f"- {source}")
        print(f"  Distance: {score:.4f}")


# ============================================================
# MAIN CHATBOT
# ============================================================

print("\nRAG Chatbot")
print("=" * 50)
print("Type 'exit' to stop.")


while True:

    question = input("\nYou: ").strip()


    # --------------------------------------------------------
    # EXIT COMMAND
    # --------------------------------------------------------

    if question.lower() == "exit":

        print("Goodbye!")
        break


    # --------------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------------

    if not question:

        print("Error: Question cannot be empty.")
        continue


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    try:

        documents = retrieve_documents(question)

    except Exception as e:

        print("\nError during retrieval.")
        print(f"Details: {e}")
        continue


    # --------------------------------------------------------
    # NO RELEVANT INFORMATION
    # --------------------------------------------------------

    if not documents:

        print("\nAssistant: I do not have enough information.")
        continue


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    try:

        answer = generate_answer(
            question,
            documents
        )

    except Exception as e:

        print("\nError generating answer.")
        print(f"Details: {e}")
        continue


    # --------------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------------

    print(f"\nAssistant: {answer}")


    # --------------------------------------------------------
    # DISPLAY SOURCES
    # --------------------------------------------------------

    display_sources(documents)
