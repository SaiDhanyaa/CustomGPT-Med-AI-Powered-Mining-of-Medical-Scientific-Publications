import torch
from transformers import BartTokenizer, BartForConditionalGeneration

def generate_summary(text, max_length=300, min_length=100):
    """
    Generate a summary for a given input text using BART.
    
    Args:
    - text: The input text to summarize (e.g., a research paper, clinical trial report).
    - max_length: Maximum length of the summary.
    - min_length: Minimum length of the summary.
    
    Returns:
    - The generated summary.
    """
    # Load the pre-trained BART model and tokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    # Tokenize the input text
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the summary (using beam search for better results)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

def summarize_document(document_path, output_path):
    """
    Summarize a research paper or clinical trial report.
    
    Args:
    - document_path: Path to the text file containing the long research paper or report.
    - output_path: Path to save the generated summary.
    
    Returns:
    - None (writes the summary to a file).
    """
    # Read the document
    with open(document_path, 'r', encoding='utf-8') as file:
        document_text = file.read()

    # Generate the summary
    summary = generate_summary(document_text)

    # Write the summary to an output file
    with open(output_path, 'w', encoding='utf-8') as summary_file:
        summary_file.write(summary)
    
    print(f"Summary saved to {output_path}")

if __name__ == "__main__":
    # Path to the research paper or clinical trial report (long text)
    document_path = 'research_paper.txt'  # Replace with the path to your document

    # Path to save the summary
    output_path = 'summary.txt'

    # Summarize the document
    summarize_document(document_path, output_path)
