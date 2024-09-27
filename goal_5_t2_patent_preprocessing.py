import re

def clean_patent_abstract(abstract):
    """
    Cleans the patent abstract by removing special characters and extra spaces.
    """
    # Remove special characters and multiple spaces
    abstract_cleaned = re.sub(r"[^A-Za-z0-9\s]", "", abstract)
    abstract_cleaned = re.sub(r"\s+", " ", abstract_cleaned).strip()
    return abstract_cleaned

# Load the fetched patents from file
with open("patent_abstracts.json", "r") as f:
    patent_abstracts = json.load(f)

# Clean each abstract
cleaned_abstracts = [clean_patent_abstract(abstract) for abstract in patent_abstracts]

# Save the cleaned abstracts to a new file
with open("cleaned_patent_abstracts.json", "w") as f:
    json.dump(cleaned_abstracts, f, indent=4)
