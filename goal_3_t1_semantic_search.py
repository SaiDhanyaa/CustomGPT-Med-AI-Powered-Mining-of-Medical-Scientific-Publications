import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Define keywords for filtering search results (optional)
cancer_keywords = ['cancer', 'tumor', 'chemotherapy', 'oncology', 'radiotherapy']

def filter_search_results(abstracts, keywords):
    """
    Filter the search results based on specific keywords.
    """
    filtered_results = []
    for abstract in abstracts:
        if any(keyword.lower() in abstract.lower() for keyword in keywords):
            filtered_results.append(abstract)
    return filtered_results

def semantic_search(query, top_k=5):
    # Load the dataset
    df = pd.read_csv('filtered_pubmed_abstracts.csv')  # Ensure this file exists

    # Initialize the pre-trained model for semantic search
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Encode both the query and the abstracts
    query_embedding = model.encode(query, convert_to_tensor=True)
    abstract_embeddings = model.encode(df['Abstract'].tolist(), convert_to_tensor=True)

    # Compute cosine similarity between the query and each abstract
    cos_scores = util.pytorch_cos_sim(query_embedding, abstract_embeddings)

    # Get the top k most relevant abstracts
    top_results = torch.topk(cos_scores, k=top_k)

    # Extract top abstracts and scores
    top_indices = top_results.indices.squeeze().tolist()  # Flatten the indices to 1D
    top_abstracts = df.iloc[top_indices]['Abstract'].tolist()
    top_scores = top_results.values.squeeze().tolist()

    # Filter results based on keywords (optional)
    filtered_abstracts = filter_search_results(top_abstracts, cancer_keywords)

    print(f"Query: {query}\n")
    print("Top filtered abstracts:")
    for idx, abstract in enumerate(filtered_abstracts[:top_k]):
        print(f"\nAbstract {idx + 1} (Score: {top_scores[idx]}):\n{abstract}\n")

# Example query
if __name__ == "__main__":
    query = "cancer treatment"  # Example query related to cancer
    semantic_search(query)
