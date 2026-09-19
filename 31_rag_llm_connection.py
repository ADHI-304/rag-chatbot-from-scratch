
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# 1. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Connect to the metadata-aware database
vectorstore = Chroma(
    persist_directory="./chroma_metadata_db",
    embedding_function=embeddings
)


# 3. Create the Ollama chat model
llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0
)


# 4. Get the user's question
question = input("Enter your question: ")


# 5. Retrieve relevant documents
documents = vectorstore.similarity_search(
    question,
    k=3
)


# 6. Combine retrieved content
context = "\n\n".join(
    document.page_content
    for document in documents
)


# 7. Create a grounded prompt
prompt = f"""
You are a helpful AI assistant.

Answer the question using only the provided context.
If the context does not contain enough information, say:
"I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:
"""


# 8. Generate the answer
response = llm.invoke(prompt)


# 9. Display the answer
print("\nGenerated Answer")
print("=" * 50)
print(response.content)
