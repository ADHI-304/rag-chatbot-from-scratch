
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Sample document
text = """
Python is a popular programming language.

Python is used in artificial intelligence and machine learning.

Machine learning allows computers to learn patterns from data.

Deep learning is a branch of machine learning.

Natural language processing works with human language.
"""


# 2. Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)


# 3. Split the text into chunks
chunks = text_splitter.split_text(text)


# 4. Display the chunks
print("Total chunks:", len(chunks))
print()

for i, chunk in enumerate(chunks, start=1):
    print(f"--- Chunk {i} ---")
    print(chunk)
    print("Characters:", len(chunk))
    print()
