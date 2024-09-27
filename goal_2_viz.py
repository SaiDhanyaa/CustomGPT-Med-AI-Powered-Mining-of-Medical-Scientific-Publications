# Import necessary libraries
import pandas as pd  # For handling and manipulating structured data
import matplotlib.pyplot as plt  # For generating plots and visualizations
from collections import Counter  # To count occurrences of elements in a list
from wordcloud import WordCloud  # For generating a word cloud visualization
import ast  # To safely evaluate strings containing Python literals (e.g., lists, dictionaries)

# Load the extracted entities from a CSV file into a pandas DataFrame
df = pd.read_csv('extracted_entities_huggingface.csv')

# Convert the 'Entities' column, which is stored as a string, back into a list of dictionaries
# ast.literal_eval safely evaluates the string and converts it back to its original Python structure
df['Entities'] = df['Entities'].apply(ast.literal_eval)

# Create an empty list to store all the extracted entities
all_entities = []

# Iterate through each row in the DataFrame to extract entities
for entities in df['Entities']:
    for entity in entities:
        # Each entity may have '##' to indicate subword tokens (used in certain tokenizers)
        # Replace '##' with an empty string to join subword tokens into complete words
        word = entity['word'].replace('##', '')
        # Append the cleaned entity to the list of all entities
        all_entities.append(word)

# Count the frequency of each entity using Counter, which returns a dictionary-like object
# where keys are the entities and values are their counts
entity_counts = Counter(all_entities)

# Step 1: Visualize the Most Common Entities using a Bar Chart
# Get the 20 most common entities and their counts
most_common_entities = entity_counts.most_common(20)

# Convert the most common entities into a pandas DataFrame for easier plotting
# The DataFrame has two columns: 'Entity' and 'Frequency'
entities_df = pd.DataFrame(most_common_entities, columns=['Entity', 'Frequency'])

# Plot a horizontal bar chart to visualize the top 20 most common entities
plt.figure(figsize=(10, 6))  # Set the figure size for better clarity
plt.barh(entities_df['Entity'], entities_df['Frequency'], color='skyblue')  # Create a horizontal bar chart
plt.xlabel('Frequency')  # Label for the x-axis
plt.title('Top 20 Most Common Biomedical Entities')  # Title of the chart
plt.gca().invert_yaxis()  # Invert the y-axis so the most common entity is at the top
plt.savefig('entity_frequency_chart_fixed.png')  # Save the bar chart as an image file
plt.show()  # Display the bar chart

# Step 2: Create a Word Cloud for the Biomedical Entities
# Generate a word cloud from the list of all entities
# ' '.join(all_entities) joins the list of entities into a single string, with each word separated by a space
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_entities))

# Display the word cloud
plt.figure(figsize=(10, 6))  # Set the figure size for better clarity
plt.imshow(wordcloud, interpolation='bilinear')  # Use bilinear interpolation for smoother word cloud rendering
plt.axis('off')  # Turn off the axis for a cleaner visualization
plt.title('Word Cloud of Biomedical Entities')  # Title of the word cloud
plt.savefig('entity_wordcloud_fixed.png')  # Save the word cloud as an image file
plt.show()  # Display the word cloud

# Output message indicating that the visualizations were successfully saved
print("Updated visualizations saved as 'entity_frequency_chart.png' and 'entity_wordcloud.png'.")
