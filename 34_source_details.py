from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# 1. Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to the original database
vectorstore = Chroma(
    persist_directory="./chroma_learning_db",
    embedding_function=embeddings
)

# 3. Load Ollama
llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0
)

# 4. Get question
question = input("Enter your question: ")

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

# 7. Generate answer
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

response = llm.invoke(prompt)

# 8. Display answer
print("\nAnswer")
print("=" * 50)
print(response.content)

# 9. Display source details
print("\nRetrieved Sources")
print("=" * 50)

for i, document in enumerate(documents, start=1):

    source = document.metadata.get(
        "source",
        "Unknown source"
    )

    print(f"\nSource {i}: {source}")
    print("Content:")
    print(document.page_content)
    print("-" * 50)
