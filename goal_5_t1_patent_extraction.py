import requests
import json
import time

# API base URL
base_url = "https://ped.uspto.gov/api/queries"

# Step 1: Define your query
def create_patent_query(search_text="COVID-19 vaccine"):
    payload = {
        "searchText": f"{search_text}",
        "fq": ["appStatus:\"Patented Case\""],
        "fl": "*",
        "facet": "false",
        "sort": "applId asc",
        "start": "0"
    }
    return payload

# Step 2: Make the POST request to create the query
def fetch_patents(query):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(base_url, headers=headers, json=query)
    
    if response.status_code == 200:
        data = response.json()
        return data['queryId'], data['searchResponse']['response']['numFound']
    else:
        print(f"Failed to fetch patents: {response.status_code}")
        return None, 0

# Step 3: Request the bundle (PUT request)
def request_bundle(query_id):
    bundle_url = f"{base_url}/{query_id}/package?format=XML"
    response = requests.put(bundle_url)
    
    if response.status_code == 200:
        return True
    else:
        print(f"Failed to package data: {response.status_code}")
        return False

# Step 4: Poll the query to check the status (GET request)
def poll_query(query_id):
    status_url = f"{base_url}/{query_id}"
    while True:
        response = requests.get(status_url)
        if response.status_code == 200:
            status = response.json()['queryStatus']
            if status == 'COMPLETED':
                return True
            else:
                print("Waiting for the package to complete...")
                time.sleep(5)
        else:
            print(f"Error polling query: {response.status_code}")
            return False

# Step 5: Download the results
def download_bundle(query_id):
    download_url = f"{base_url}/{query_id}/download"
    response = requests.get(download_url)
    
    if response.status_code == 200:
        with open(f'patents_{query_id}.zip', 'wb') as file:
            file.write(response.content)
        print(f"Downloaded bundle for query: {query_id}")
    else:
        print(f"Failed to download bundle: {response.status_code}")

# Main workflow
if __name__ == "__main__":
    search_query = create_patent_query("COVID-19 vaccine")
    query_id, num_found = fetch_patents(search_query)
    
    if query_id and num_found > 0:
        print(f"Found {num_found} patents.")
        
        if request_bundle(query_id):
            if poll_query(query_id):
                download_bundle(query_id)
    else:
        print("No patents found.")
