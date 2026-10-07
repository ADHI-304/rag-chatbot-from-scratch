from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# 1. Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to Chroma
vectorstore = Chroma(
    persist_directory="./chroma_learning_db",
    embedding_function=embeddings
)

# 3. Load Ollama
llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0
)

print("\nRAG Chatbot")
print("=" * 50)
print("Type 'exit' to stop the chatbot.")

# 4. Chat loop
while True:

    question = input("\nYou: ")

    # Exit condition
    if question.lower() == "exit":
        print("Goodbye!")
        break

    # 5. Retrieve documents
    documents = vectorstore.similarity_search(
        question,
        k=3
    )

    # 6. Create context
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # 7. Create prompt
    prompt = f"""
You are a reliable question-answering assistant.

Answer using only the provided context.

If the context does not contain enough information,
say that you do not have enough information.

Context:
{context}

Question:
{question}

Answer:
"""

    # 8. Generate answer
    response = llm.invoke(prompt)

    # 9. Display answer
    print("\nAssistant:", response.content)
