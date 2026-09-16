
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. Read the knowledge file
with open("knowledge.txt", "r", encoding="utf-8") as file:
    text = file.read()


# 2. Create a LangChain Document
document = Document(
    page_content=text,
    metadata={
        "source": "knowledge.txt"
    }
)


# 3. Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30
)

chunks = text_splitter.split_documents([document])

print("Total chunks:", len(chunks))


# 4. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 5. Create the Chroma vector database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_learning_db"
)


# 6. Display success message
print("Documents stored successfully!")
print("Chroma database created at: ./chroma_learning_db")
