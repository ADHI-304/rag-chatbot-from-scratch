
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


# 1. Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Connect to Chroma
vectorstore = Chroma(
    persist_directory="./chroma_learning_db",
    embedding_function=embeddings
)


# 3. Create LLM
llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0
)


# 4. Define similarity threshold
THRESHOLD = 1.3


# 5. Create prompt template
prompt_template = PromptTemplate.from_template(
    """
    You are a helpful AI assistant.

    Answer ONLY using the provided context.

    If the context does not contain enough information,
    say: I do not have enough information.

    Do not use outside knowledge.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)


# 6. Get user question
query = input("You: ")


# 7. Retrieve documents with scores
results = vectorstore.similarity_search_with_score(
    query,
    k=2
)


# 8. Filter documents using threshold
relevant_documents = [
    document
    for document, score in results
    if score <= THRESHOLD
]


# 9. Check whether relevant documents exist
if not relevant_documents:

    print("\nBot: I do not have enough information.")
    print("\nNo sufficiently similar documents found.")

else:

    # 10. Combine relevant documents
    context = "\n\n".join(
        document.page_content
        for document in relevant_documents
    )

    # 11. Format prompt
    prompt = prompt_template.format(
        context=context,
        question=query
    )

    # 12. Generate answer
    response = llm.invoke(prompt)

    # 13. Display answer
    print("\nBot:", response.content)

    # 14. Display sources
    print("\nSources:")

    sources = set()

    for document in relevant_documents:
        source = document.metadata.get(
            "source",
            "Unknown"
        )

        sources.add(source)

    for source in sources:
        print(f"- {source}")
