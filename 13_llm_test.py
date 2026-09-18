
from langchain_ollama import ChatOllama


# 1. Create the LLM
llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0
)


# 2. Send a question
response = llm.invoke(
    "What is machine learning? Explain in simple words."
)


# 3. Display the response
print("LLM Response:")
print(response.content)
