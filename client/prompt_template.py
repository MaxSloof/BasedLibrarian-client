from langchain.prompts import PromptTemplate

# Custom propmt
PROMPT_TEMPLATE = """Given the following extracted parts of a long document and a question, give a detailed response to the question if you know the answer.
If you don't know the answer, just say that you don't know. Don't try to make up an answer. Also do not provide an answer if no extracted parts have been given.

QUESTION: {question}
=========
{summaries}
=========
Answer: Let's think step by step."""
QUESTION_PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question","summaries"])