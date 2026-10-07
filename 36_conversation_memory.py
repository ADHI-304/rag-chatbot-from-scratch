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
# 4. CONVERSATION MEMORY
# ============================================================

conversation_history = []


print("\nRAG Chatbot with Memory")
print("=" * 50)
print("Type 'exit' to stop.")


# ============================================================
# 5. CHAT LOOP
# ============================================================

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break


    # ========================================================
    # 6. REWRITE FOLLOW-UP QUESTION
    # ========================================================

    if conversation_history:

        previous_question = conversation_history[-1][0]
        previous_answer = conversation_history[-1][1]

        rewrite_prompt = f"""
Rewrite the latest question so that it is a complete,
standalone question.

Previous question:
{previous_question}

Previous answer:
{previous_answer}

Latest question:
{question}

Return ONLY the rewritten question.
"""

        rewrite_response = llm.invoke(rewrite_prompt)

        retrieval_query = rewrite_response.content.strip()

    else:

        retrieval_query = question


    print("\n[Debug] Retrieval query:")
    print(retrieval_query)


    # ========================================================
    # 7. RETRIEVE DOCUMENTS
    # ========================================================

    documents = vectorstore.similarity_search(
        retrieval_query,
        k=4
    )


    # ========================================================
    # 8. CREATE CONTEXT
    # ========================================================

    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    # ========================================================
    # 9. CREATE CONVERSATION HISTORY
    # ========================================================

    history = "\n".join(
        f"User: {user}\nAssistant: {assistant}"
        for user, assistant in conversation_history
    )


    # ========================================================
    # 10. FINAL ANSWER
    # ========================================================

    prompt = f"""
Answer the user's question using the information below.

Retrieved information:
{context}

Previous conversation:
{history}

Question:
{question}

If the retrieved information contains the answer,
give the answer directly.

Do not say that information is missing if the answer
is clearly present in the retrieved information.

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

    print("\nAssistant:", answer)


    # ========================================================
    # 13. SAVE MEMORY
    # ========================================================

    conversation_history.append(
        (question, answer)
    )
