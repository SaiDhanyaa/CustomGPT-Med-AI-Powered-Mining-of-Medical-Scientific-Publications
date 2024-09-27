from Bio import Entrez  # Import the Entrez module from Biopython for accessing PubMed
import pandas as pd  # Import pandas for data manipulation and saving results in CSV format

# Define your email for PubMed API access (this is required by NCBI PubMed)
Entrez.email = "your_email@example.com"  # Replace with your email to identify yourself for the API

# Function to fetch data from PubMed
def fetch_pubmed_data(query, max_results=100):
    """
    Fetch abstracts from PubMed based on a search query.

    Args:
    - query (str): Search term for PubMed (e.g., "cancer", "COVID-19").
    - max_results (int): The maximum number of abstracts to retrieve (default is 100).

    Returns:
    - abstracts (list): A list containing the fetched abstracts.
    """
    
    # Step 1: Search PubMed using the specified query and maximum results
    print(f"Fetching {max_results} PubMed articles for query: {query}")
    
    # Use Entrez.esearch to search PubMed. 'term' is the search query, 'retmax' limits the number of results.
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    
    # Read and parse the search results
    record = Entrez.read(handle)
    handle.close()  # Always close the handle after reading the results
    
    # Step 2: Extract the PubMed IDs from the search results
    id_list = record["IdList"]  # 'IdList' contains the unique PubMed IDs of the search results
    print(f"Found {len(id_list)} articles.")  # Print the number of articles found

    # Step 3: Use Entrez.efetch to retrieve detailed information (abstracts) based on the PubMed IDs
    # 'rettype="abstract"' specifies that we want the abstracts in text form, 'retmode="text"' indicates plain text.
    handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="abstract", retmode="text")
    
    # Read the abstracts fetched
    abstracts = handle.read()
    handle.close()  # Close the handle after fetching the abstracts

    # Step 4: Return the fetched abstracts as a list, splitting by newlines ("\n\n")
    # Each abstract is separated by two newlines, so splitting this way provides individual abstracts.
    return abstracts.split("\n\n")

# Function to save fetched data to a CSV file
def save_data_to_csv(abstracts, filename):
    """
    Save the fetched abstracts into a CSV file.

    Args:
    - abstracts (list): List of abstracts to save.
    - filename (str): The name of the CSV file where data will be saved.

    Returns:
    - None
    """
    
    # Convert the list of abstracts into a pandas DataFrame
    df = pd.DataFrame(abstracts, columns=["Abstract"])
    
    # Save the DataFrame to a CSV file without row indices
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")  # Confirm that data has been saved

# Main execution: Fetch and save PubMed abstracts
if __name__ == "__main__":
    """
    Example usage:
    This section will execute if the script is run as a standalone file. It demonstrates
    how to fetch and save data from PubMed using the above functions.
    """
    
    # Specify the search query, e.g., a medical term such as "COVID-19"
    query = "COVID-19"  # You can modify this query to search for different topics
    
    # Fetch PubMed abstracts using the defined query and a limit of 100 articles
    abstracts = fetch_pubmed_data(query, max_results=100)

    # Save the fetched abstracts to a CSV file named 'pubmed_abstracts.csv'
    save_data_to_csv(abstracts, "pubmed_abstracts.csv")
