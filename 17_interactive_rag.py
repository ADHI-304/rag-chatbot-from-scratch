
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


# 4. Create the RAG prompt template
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


# 5. Display chatbot heading
print("===================================")
print("       RAG CHATBOT")
print("===================================")
print("Type 'exit' to quit.\n")


# 6. Start the chatbot loop
while True:

    # Get user input
    query = input("You: ")

    # Exit condition
    if query.lower() == "exit":
        print("Goodbye!")
        break

    # Ignore empty questions
    if not query.strip():
        print("Please enter a question.\n")
        continue

    # Retrieve relevant documents
    documents = vectorstore.similarity_search(
        query,
        k=2
    )

    # Combine retrieved documents into context
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Fill the prompt template
    prompt = prompt_template.format(
        context=context,
        question=query
    )

    # Generate the answer
    response = llm.invoke(prompt)

    # Display the answer
    print("\nBot:", response.content)
    print("\n" + "-" * 40 + "\n")
