
from langchain_core.prompts import PromptTemplate


# 1. Create a reusable prompt template
template = PromptTemplate.from_template(
    """
    You are a helpful AI assistant.

    Explain the following topic in simple words.

    Topic: {topic}
    """
)


# 2. Fill the template with a topic
prompt = template.format(
    topic="Artificial Intelligence"
)


# 3. Display the generated prompt
print("Generated Prompt:")
print(prompt)
