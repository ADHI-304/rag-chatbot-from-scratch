
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


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


# 4. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 5. Generate embeddings for each chunk
vectors = embeddings.embed_documents(
    [chunk.page_content for chunk in chunks]
)


# 6. Display the results
print("Total chunks:", len(chunks))
print("Total vectors:", len(vectors))

print()

for i, vector in enumerate(vectors, start=1):
    print(f"Chunk {i}")
    print("Vector length:", len(vector))
    print("First 5 values:", vector[:5])
    print()
