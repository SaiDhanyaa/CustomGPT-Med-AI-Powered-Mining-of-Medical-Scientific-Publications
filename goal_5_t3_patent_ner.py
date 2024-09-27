from transformers import pipeline

# Load the pre-trained NER model for biomedical data
ner_model = pipeline("ner", model="d4data/biomedical-ner-all")

# Load the cleaned abstracts
with open("cleaned_patent_abstracts.json", "r") as f:
    cleaned_abstracts = json.load(f)

# Perform NER on the cleaned abstracts
extracted_entities = []
for abstract in cleaned_abstracts:
    entities = ner_model(abstract)
    extracted_entities.append(entities)

# Save the extracted entities to a file
with open("extracted_patent_entities.json", "w") as f:
    json.dump(extracted_entities, f, indent=4)

print("Named entities extracted from patents saved to 'extracted_patent_entities.json'.")
