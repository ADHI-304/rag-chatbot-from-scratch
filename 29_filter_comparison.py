
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = Chroma(
    persist_directory="./chroma_metadata_db",
    embedding_function=embeddings
)


question = input("Enter your question: ")


# Unfiltered retrieval
normal_results = vectorstore.similarity_search(
    question,
    k=3
)


# Filtered retrieval
filtered_results = vectorstore.similarity_search(
    question,
    k=3,
    filter={"category": "AI"}
)


print("\nUNFILTERED RESULTS")
print("=" * 50)

for i, document in enumerate(normal_results, start=1):
    print(f"\nResult {i}")
    print("Content:", document.page_content)
    print("Category:", document.metadata.get("category"))


print("\nFILTERED RESULTS (AI ONLY)")
print("=" * 50)

for i, document in enumerate(filtered_results, start=1):
    print(f"\nResult {i}")
    print("Content:", document.page_content)
    print("Category:", document.metadata.get("category"))
