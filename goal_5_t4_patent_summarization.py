from transformers import BartTokenizer, BartForConditionalGeneration

# Load the BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Load the cleaned abstracts
with open("cleaned_patent_abstracts.json", "r") as f:
    cleaned_abstracts = json.load(f)

# Summarize each abstract
summarized_abstracts = []
for abstract in cleaned_abstracts:
    inputs = tokenizer([abstract], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=200, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summarized_abstracts.append(summary)

# Save the summaries to a file
with open("patent_summaries.json", "w") as f:
    json.dump(summarized_abstracts, f, indent=4)

print("Summaries of patent abstracts saved to 'patent_summaries.json'.")
