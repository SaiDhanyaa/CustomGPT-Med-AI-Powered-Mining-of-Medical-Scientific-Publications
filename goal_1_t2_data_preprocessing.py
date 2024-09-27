import pandas as pd

# Define keywords for filtering abstracts related to cancer
cancer_keywords = ['cancer', 'tumor', 'chemotherapy', 'oncology', 'radiotherapy']

def filter_abstracts(abstracts, keywords):
    """
    Filter abstracts based on the presence of certain keywords.
    """
    filtered_abstracts = []
    for abstract in abstracts:
        if any(keyword.lower() in abstract.lower() for keyword in keywords):
            filtered_abstracts.append(abstract)
    return filtered_abstracts

if __name__ == "__main__":
    # Load abstracts from the dataset
    df = pd.read_csv('pubmed_abstracts.csv')  # Load your abstract data

    # Filter abstracts related to cancer
    filtered_abstracts = filter_abstracts(df['Abstract'], cancer_keywords)

    # Save the filtered abstracts
    pd.DataFrame(filtered_abstracts, columns=['Abstract']).to_csv('filtered_pubmed_abstracts.csv', index=False)
    print(f"Filtered {len(filtered_abstracts)} abstracts related to cancer.")
