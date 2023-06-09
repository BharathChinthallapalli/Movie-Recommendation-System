# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from difflib import get_close_matches

# Load data into DataFrame
netflix_df = pd.read_csv("titles.csv")

# Convert 'description' column to string
netflix_df['description'] = netflix_df['description'].astype(str)

# Preprocessing and feature engineering
netflix_df['combined_features'] = netflix_df['title'] + ' ' + netflix_df['type'] + ' ' + ['production_countries'] + ' ' + netflix_df['description']

# Create Doc2Vec model
tagged_data = [TaggedDocument(words=movie.split(), tags=[str(i)]) for i, movie in enumerate(netflix_df['combined_features'])]
d2v_model = Doc2Vec(tagged_data, vector_size=100, window=5, min_count=1, workers=4)

# Perform dimensionality reduction
vector_matrix = np.array([d2v_model.infer_vector(movie.split()) for movie in netflix_df['combined_features']])
svd_model = TruncatedSVD(n_components=50)
reduced_vectors = svd_model.fit_transform(vector_matrix)

# Perform K-means clustering
kmeans_model = KMeans(n_clusters=10, init='k-means++', max_iter=100, n_init=1)
kmeans_model.fit(reduced_vectors)
netflix_df['cluster'] = kmeans_model.labels_

# Function to recommend similar movies
def recommend_movies(movie_title, similarity_matrix, top_n=5):
    closest_match = get_close_matches(movie_title, netflix_df['title'], n=1, cutoff=0.8)
    if closest_match:
        closest_match = closest_match[0]
        cluster = netflix_df[netflix_df['title'] == closest_match]['cluster'].values[0]
        cluster_movies = netflix_df[netflix_df['cluster'] == cluster]
        similarity_scores = cosine_similarity([d2v_model.infer_vector(movie.split()) for movie in cluster_movies['combined_features']])
        similarity_scores = similarity_scores[:, 0]
        similar_movies_indices = similarity_scores.argsort()[::-1][1:top_n+1]
        similar_movies = cluster_movies.iloc[similar_movies_indices]['title']
        return similar_movies
    else:
        print("No similar movies found.")
        return None

if __name__ == '__main__':
    movie_title = input("Enter the movie title: ")
    similar_movies = recommend_movies(movie_title, cosine_similarity, top_n=5)
    if similar_movies is not None:
        print("Similar Movies:")
        print(similar_movies)

