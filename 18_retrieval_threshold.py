
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = Chroma(
    persist_directory="./chroma_learning_db",
    embedding_function=embeddings
)


query = "What is cricket?"


results = vectorstore.similarity_search_with_score(
    query,
    k=4
)


for i, (document, score) in enumerate(results, start=1):

    print(f"\nResult {i}")
    print(f"Distance: {score:.4f}")
    print(f"Content: {document.page_content}")

