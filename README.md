# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
movies = pd.read_csv("movies.csv")

# Data Preprocessing
movies["overview"].fillna("", inplace=True)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["overview"])

# Compute Similarity Matrix
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation Function
def recommend_movie(title, num_recommendations=5):
    idx = movies[movies["title"] == title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:num_recommendations+1]
    
    movie_indices = [i[0] for i in scores]
    return movies["title"].iloc[movie_indices]

# Example Usage
print(recommend_movie("The Dark Knight"))# Movie-Recommendation-System
