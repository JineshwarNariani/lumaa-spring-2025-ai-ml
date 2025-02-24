import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

def load_data(file_path):
    # Read CSV into a DataFrame
    df = pd.read_csv(file_path)
    print("Columns in the dataset:", df.columns)  # Debug: Print column names
    
    # Use the 'Description' column (adjust if needed) and convert to lowercase
    if 'Description' in df.columns:
        df['Description'] = df['Description'].fillna("").str.lower()
    else:
        raise ValueError("The dataset does not contain a 'Description' column.")
    
    return df

# Load the dataset (adjust the path as needed)
data = load_data("IMDB-Movie-Data.csv")
print(data.head())

# Build the TF-IDF matrix using the 'Description' column instead of 'plot'
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['Description'])
print("TF-IDF matrix shape:", tfidf_matrix.shape)


def recommend_items(user_query, tfidf_matrix, data, top_n=5):
    # Transform the user query into the same TF-IDF space
    query_vec = vectorizer.transform([user_query.lower()])
    # Compute cosine similarity between the query and each item
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # Get the indices of the top N most similar items
    top_indices = np.argsort(cosine_sim)[::-1][:top_n]
    # Retrieve the corresponding items
    recommendations = data.iloc[top_indices].copy()
    recommendations['similarity'] = cosine_sim[top_indices]
    return recommendations

if __name__ == "__main__":
    # If a command-line argument is provided, use it; otherwise, use the default query.
    query = sys.argv[1] if len(sys.argv) > 1 else "I love thrilling action movies set in space, with a comedic twist."
    recommended_items = recommend_items(query, tfidf_matrix, data, top_n=5)
    print("Recommendations:")
    print(recommended_items[['Title', 'similarity']])
