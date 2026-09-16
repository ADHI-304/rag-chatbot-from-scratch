
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


# 3. Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30
)


# 4. Split the document into chunks
chunks = text_splitter.split_documents([document])


# 5. Display the chunks
print("Document loaded and split successfully!")
print("Total chunks:", len(chunks))
print()

for i, chunk in enumerate(chunks, start=1):

    print(f"--- Chunk {i} ---")
    print("Content:", chunk.page_content)
    print("Metadata:", chunk.metadata)
    print("Characters:", len(chunk.page_content))
    print()
