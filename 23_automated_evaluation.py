
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Connect to the existing vector database
vectorstore = Chroma(
    persist_directory="./chroma_learning_db",
    embedding_function=embeddings
)


# 3. Test questions and expected keywords
test_questions = [
    {
        "question": "What is machine learning?",
        "keyword": "learn"
    },
    {
        "question": "What is deep learning?",
        "keyword": "neural"
    },
    {
        "question": "What is natural language processing?",
        "keyword": "language"
    },
    {
        "question": "What is RAG?",
        "keyword": "retrieval"
    }
]


successful_retrievals = 0


# 4. Evaluate each question
for test in test_questions:

    question = test["question"]
    keyword = test["keyword"].lower()

    results = vectorstore.similarity_search(
        question,
        k=3
    )

    combined_content = ""

    for document in results:
        combined_content += document.page_content.lower()

    is_successful = keyword in combined_content

    print("\nQuestion:", question)
    print("Expected keyword:", keyword)
    print("Retrieval successful:", is_successful)

    if is_successful:
        successful_retrievals += 1


# 5. Calculate success rate
total_questions = len(test_questions)

success_rate = (
    successful_retrievals / total_questions
) * 100


print("\n" + "=" * 40)
print("Retrieval Evaluation Summary")
print("=" * 40)

print("Successful retrievals:", successful_retrievals)
print("Total questions:", total_questions)
print(f"Success rate: {success_rate:.2f}%")
