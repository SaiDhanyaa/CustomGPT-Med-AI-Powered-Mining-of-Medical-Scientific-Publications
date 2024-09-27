from transformers import pipeline

# Load pre-trained QA model from Hugging Face
qa_model = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

# Sample context text (this can be an abstract from a paper)
context = """
COVID-19 vaccines have been shown to reduce the risk of severe illness, hospitalization, and death. Clinical trials have demonstrated that vaccines provide strong protection against the disease.
"""

# Function to perform question-answering
def answer_question(question, context):
    """
    Given a question and context, return the answer based on the context.
    
    Arguments:
    - question: A question string
    - context: A string of text to search the answer from
    
    Returns:
    - Answer to the question
    """
    result = qa_model(question=question, context=context)
    return result['answer']

# Example usage:
if __name__ == "__main__":
    question = "What do COVID-19 vaccines reduce the risk of?"
    answer = answer_question(question, context)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
