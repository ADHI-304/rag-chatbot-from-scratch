
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Create a LangChain Document
document = Document(
    page_content="""
    Python is a popular programming language.
    Python is used in artificial intelligence.
    Machine learning allows computers to learn patterns from data.
    """,
    metadata={
        "source": "knowledge.txt",
        "topic": "AI"
    }
)


# 2. Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)


# 3. Split the Document into smaller Documents
chunks = text_splitter.split_documents([document])


# 4. Display the chunks
print("Total chunks:", len(chunks))
print()

for i, chunk in enumerate(chunks, start=1):
    print(f"--- Chunk {i} ---")
    print("Content:", chunk.page_content)
    print("Metadata:", chunk.metadata)
    print()
