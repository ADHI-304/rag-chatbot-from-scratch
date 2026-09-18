
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# 1. Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to Chroma database
vectorstore = Chroma(
    persist_directory="./chroma_learning_db",
    embedding_function=embeddings
)

# 3. Load Ollama model
llm = OllamaLLM(model="qwen2.5-coder:7b")

# 4. Create grounded prompt
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Answer the question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the context does not contain the answer, say:
   "I do not have enough information in my knowledge base."

Context:
{context}

Question:
{question}

Answer:
"""
)

# 5. Get user question
query = input("Enter your question: ")

# 6. Retrieve relevant documents
results = vectorstore.similarity_search_with_score(
    query,
    k=4
)

# 7. Apply retrieval threshold
threshold = 0.6

relevant_docs = []

for document, distance in results:

    if distance <= threshold:
        relevant_docs.append(document)

# 8. Handle no relevant documents
if not relevant_docs:

    print("\nBot: I do not have enough information in my knowledge base.")

else:

    # 9. Combine retrieved content
    context = "\n\n".join(
        document.page_content
        for document in relevant_docs
    )

    # 10. Build the prompt
    formatted_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # 11. Generate answer
    response = llm.invoke(formatted_prompt)

    print("\nBot:", response)
