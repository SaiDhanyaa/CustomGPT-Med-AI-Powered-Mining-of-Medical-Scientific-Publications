import gradio as gr
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

# Load models
semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
qa_model = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

# Example abstracts (you can replace this with actual data from your search)
abstracts = [
    "COVID-19 vaccines have been shown to reduce the risk of severe illness, hospitalization, and death.",
    "The mRNA vaccines are highly effective but come with a higher incidence of side effects.",
    "The efficacy of COVID-19 vaccines has been demonstrated in multiple clinical trials.",
    "New variants of COVID-19 may affect the long-term efficacy of vaccines.",
    "Vaccines play a critical role in controlling the spread of infectious diseases."
]

# Semantic search function
def semantic_search(query):
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    corpus_embeddings = semantic_model.encode(abstracts, convert_to_tensor=True)
    
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=3)  # Get top 3 results
    
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append((abstracts[idx], score.item()))
    
    return results

# Question-answering function
def question_answering(question, context):
    result = qa_model(question=question, context=context)
    return result['answer']

# Gradio Interface
def interactive_app(query, question):
    search_results = semantic_search(query)
    
    # Return the top result's abstract and answer the user's question
    top_abstract = search_results[0][0]
    answer = question_answering(question, top_abstract)
    
    return f"Top Abstract: {top_abstract}\n\nAnswer to your question: {answer}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# CustomGPT: Semantic Search and Question Answering")
    
    with gr.Row():
        query_input = gr.Textbox(label="Enter your search query")
        question_input = gr.Textbox(label="Ask a question based on the top result")

    submit_btn = gr.Button("Submit")
    output_box = gr.Textbox(label="Results")

    submit_btn.click(interactive_app, inputs=[query_input, question_input], outputs=output_box)

# Launch the app
demo.launch()
