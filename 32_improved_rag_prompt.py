
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# 1. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Connect to the vector database
vectorstore = Chroma(
    persist_directory="./chroma_metadata_db",
    embedding_function=embeddings
)


# 3. Initialize the LLM
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


# 6. Create the context
context = "\n\n".join(
    document.page_content
    for document in documents
)


# 7. Improved prompt
prompt = f"""
You are a reliable question-answering assistant.

Follow these rules carefully:

1. Answer using only the provided context.
2. Do not invent facts or information.
3. Give a clear and concise answer.
4. If the context does not contain enough information,
   say that you do not have enough information.
5. Do not assume that information outside the context
   is correct.

Context:
{context}

User Question:
{question}

Answer:
"""


# 8. Generate the response
response = llm.invoke(prompt)


# 9. Display the answer
print("\nImproved RAG Answer")
print("=" * 50)
print(response.content)
