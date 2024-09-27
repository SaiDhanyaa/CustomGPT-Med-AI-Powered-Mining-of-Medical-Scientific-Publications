from transformers import pipeline
import pandas as pd

# Load BioBERT or any other Hugging Face NER model for biomedical data
# You can replace 'd4data/biomedical-ner-all' with another model if needed
# This model is pre-trained on biomedical texts and optimized for extracting entities like diseases, drugs, genes, etc.
nlp = pipeline("ner", model="d4data/biomedical-ner-all")

def extract_entities(text):
    """
    Extract named entities using Hugging Face NER model.
    Args:
        text (str): The text to process (scientific publication abstract).
    Returns:
        entities (dict): Extracted entities categorized by type.
    """
    # Run the Named Entity Recognition (NER) pipeline on the input text
    # The 'nlp' model processes the text and returns a list of recognized entities with associated metadata
    ner_results = nlp(text)
    
    # Initialize a dictionary to store the entities extracted from the text
    entities = {'entities': []}

    # Iterate over each result produced by the NER model
    # Each 'result' contains:
    # - 'word': The word or token recognized as an entity
    # - 'entity': The category/type of the entity (e.g., drug, disease, gene)
    # - 'score': The confidence score of the recognition
    for result in ner_results:
        entities['entities'].append({
            'word': result['word'],   # The extracted word
            'entity': result['entity'],  # The entity type (e.g., B-Disease, I-Drug)
            'score': result['score']   # Confidence score of the entity extraction
        })
    
    return entities  # Return the dictionary containing extracted entities

def process_publications(input_file, output_file):
    """
    Process each publication's abstract and extract entities, then save results.
    Args:
        input_file (str): Path to the cleaned abstracts CSV file.
        output_file (str): Path to save the extracted entities in a CSV file.
    """
    # Load the dataset containing scientific publication abstracts
    # The dataset is expected to be a CSV file where one column contains cleaned abstracts
    df = pd.read_csv(input_file)

    # Initialize a list to store extracted entities for each abstract
    results = []

    # Iterate over each row (each publication) in the dataset
    for _, row in df.iterrows():
        abstract = row["Cleaned_Abstract"]  # Retrieve the abstract from the dataset

        # Extract entities from the abstract using the previously defined 'extract_entities' function
        entities = extract_entities(abstract)

        # Append the extracted entities to the results list
        results.append({
            'Abstract': abstract,      # The abstract text
            'Entities': entities['entities']  # List of extracted entities
        })

    # Convert the results (list of dictionaries) into a pandas DataFrame
    # Each row contains the abstract and its corresponding extracted entities
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file, where each row contains the abstract and its extracted entities
    results_df.to_csv(output_file, index=False)

    # Notify the user that the entity extraction process is completed and the results are saved
    print(f"Entity extraction completed. Results saved to {output_file}")

if __name__ == "__main__":
    # Example usage of the pipeline
    # Define the input file containing cleaned abstracts
    input_file = "cleaned_pubmed_abstracts.csv"  # Cleaned abstracts from your dataset

    # Define the output file where extracted entities will be saved
    output_file = "extracted_entities_huggingface.csv"  # Output file with the entities

    # Process the publications and extract entities
    process_publications(input_file, output_file)
