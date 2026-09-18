
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


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


# 4. Create the prompt template
prompt_template = PromptTemplate.from_template(
    """
    You are a helpful AI assistant.

    Answer the question using only the provided context.

    If the answer is not present in the context,
    say that you do not have enough information.

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
documents = vectorstore.similarity_search(
    query,
    k=2
)


# 7. Combine document contents into context
context = "\n\n".join(
    document.page_content
    for document in documents
)


# 8. Fill the prompt template
prompt = prompt_template.format(
    context=context,
    question=query
)


# 9. Generate the answer
response = llm.invoke(prompt)


# 10. Display the generated answer
print("\nGenerated Answer:")
print(response.content)


# 11. Display source citations
print("\nSources:")

sources = set()

for document in documents:
    source = document.metadata.get("source", "Unknown")
    sources.add(source)

for source in sources:
    print(f"- {source}")
