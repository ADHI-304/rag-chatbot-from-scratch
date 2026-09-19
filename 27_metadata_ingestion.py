
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. Create documents with metadata
documents = [
    Document(
        page_content="Machine learning allows computers to learn patterns from data.",
        metadata={"category": "AI"}
    ),
    Document(
        page_content="Deep learning uses neural networks to learn complex patterns.",
        metadata={"category": "AI"}
    ),
    Document(
        page_content="Python is a popular programming language.",
        metadata={"category": "Programming"}
    ),
    Document(
        page_content="RAG combines document retrieval with language generation.",
        metadata={"category": "Generative AI"}
    )
]


# 2. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 3. Create a separate vector database
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_metadata_db"
)


print("Metadata-aware database created successfully.")
print("Total documents:", len(documents))

for document in documents:
    print("\nContent:", document.page_content)
    print("Metadata:", document.metadata)
