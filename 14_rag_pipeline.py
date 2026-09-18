
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# 1. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Connect to the existing Chroma database
vectorstore = Chroma(
    persist_directory="./chroma_learning_db",
    embedding_function=embeddings
)


# 3. Create the LLM
llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0
)


# 4. Ask a question
query = "What is machine learning?"


# 5. Retrieve relevant documents
documents = vectorstore.similarity_search(
    query,
    k=2
)


# 6. Combine retrieved documents into context
context = "\n\n".join(
    document.page_content
    for document in documents
)


# 7. Create the prompt
prompt = f"""
You are a helpful AI assistant.

Answer the question using the provided context.

If the answer is not present in the context,
say that you do not have enough information.

Context:
{context}

Question:
{query}

Answer:
"""


# 8. Generate the answer
response = llm.invoke(prompt)


# 9. Display the result
print("Question:", query)
print("\nRetrieved Context:")
print(context)

print("\nGenerated Answer:")
print(response.content)
