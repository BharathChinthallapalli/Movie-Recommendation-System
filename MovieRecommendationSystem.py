# -*- coding: utf-8 -*-
"""
Movie Recommendation System

This script implements a content-based movie recommendation system.
It processes a dataset of movies, generates vector embeddings for them using Doc2Vec,
clusters similar movies using K-Means, and then recommends movies based on
user input.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from difflib import get_close_matches
from typing import List, Optional, Tuple

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Loads movie data from a CSV file into a pandas DataFrame.

    Args:
        csv_path (str): The file path to the CSV data.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        raise
    except Exception as e:
        print(f"Error loading data from {csv_path}: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs preprocessing on the DataFrame.

    Converts 'description' to string and creates 'combined_features'
    by concatenating 'title', 'type', 'production_countries', and 'description'.
    Ensures all features are strings and NaNs are filled with empty strings.

    Args:
        df (pd.DataFrame): The raw DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the new 'combined_features' column.
    """
    df_processed = df.copy()
    # Convert 'description' column to string, fill NaNs
    df_processed['description'] = df_processed['description'].fillna('').astype(str)
    
    # Ensure all relevant columns are strings and fill NaNs before concatenation
    df_processed['title'] = df_processed['title'].fillna('').astype(str)
    df_processed['type'] = df_processed['type'].fillna('').astype(str)
    df_processed['production_countries'] = df_processed['production_countries'].fillna('').astype(str)

    # Create combined_features
    df_processed['combined_features'] = (
        df_processed['title'] + ' ' +
        df_processed['type'] + ' ' +
        df_processed['production_countries'] + ' ' +
        df_processed['description']
    )
    return df_processed

def train_doc2vec_model(
    df: pd.DataFrame,
    column_name: str = 'combined_features',
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    workers: int = 4,
    epochs: int = 20 # Added epochs for better training
) -> Doc2Vec:
    """
    Trains a Doc2Vec model on the specified text column in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        column_name (str): Name of the column with text for training.
        vector_size (int): Dimensionality of the feature vectors.
        window (int): Maximum distance between the current and predicted word.
        min_count (int): Ignores all words with total frequency lower than this.
        workers (int): Number of worker threads to train the model.
        epochs (int): Number of iterations (epochs) over the corpus.

    Returns:
        Doc2Vec: The trained Doc2Vec model.
    """
    tagged_data = [
        TaggedDocument(words=movie_text.split(), tags=[str(i)])
        for i, movie_text in enumerate(df[column_name])
    ]
    # Instantiate the model
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        dm=1 # Using PV-DM algorithm
    )
    # Build vocabulary
    model.build_vocab(tagged_data)
    # Train the model
    model.train(tagged_data, total_examples=model.corpus_count, epochs=epochs)
    return model

def apply_dimensionality_reduction(
    d2v_model: Doc2Vec,
    df: pd.DataFrame,
    column_name: str = 'combined_features',
    n_components: int = 50
) -> np.ndarray:
    """
    Applies TruncatedSVD to reduce the dimensionality of Doc2Vec vectors.

    Args:
        d2v_model (Doc2Vec): The trained Doc2Vec model.
        df (pd.DataFrame): DataFrame containing the text data in column_name.
        column_name (str): Name of the column used to infer vectors.
        n_components (int): Desired dimensionality of the output vectors.

    Returns:
        np.ndarray: Matrix of reduced feature vectors.
    """
    # Infer vectors for each document in the specified column
    vector_matrix = np.array(
        [d2v_model.infer_vector(text.split()) for text in df[column_name]]
    )
    
    # Apply TruncatedSVD
    svd_model = TruncatedSVD(n_components=n_components, random_state=42) # Added random_state for reproducibility
    reduced_vectors = svd_model.fit_transform(vector_matrix)
    return reduced_vectors

def perform_clustering(
    vectors: np.ndarray,
    n_clusters: int = 10,
    init: str = 'k-means++',
    max_iter: int = 300, # Default sklearn max_iter
    n_init: int = 10     # Default sklearn n_init
) -> Tuple[KMeans, np.ndarray]:
    """
    Performs K-Means clustering on the provided vectors.

    Args:
        vectors (np.ndarray): Matrix of feature vectors to cluster.
        n_clusters (int): The number of clusters to form.
        init (str): Method for initialization ('k-means++' or 'random').
        max_iter (int): Maximum number of iterations for a single run.
        n_init (int): Number of times k-means will be run with different seeds.

    Returns:
        Tuple[KMeans, np.ndarray]: The trained KMeans model and the cluster labels for each sample.
    """
    kmeans_model = KMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        n_init=n_init,
        random_state=42 # Added random_state for reproducibility
    )
    kmeans_model.fit(vectors)
    labels = kmeans_model.labels_
    return kmeans_model, labels

def get_recommendations(
    movie_title: str,
    df: pd.DataFrame, # Expects DataFrame with 'title' and 'cluster' columns
    d2v_model: Doc2Vec, # The trained Doc2Vec model
    top_n: int = 5,
    similarity_cutoff: float = 0.6 # Stricter cutoff for closer matches
) -> Optional[List[str]]:
    """
    Recommends similar movies based on a given movie title.

    Finds the closest match for the movie title, identifies its cluster,
    and then calculates cosine similarity for movies within the same cluster
    using their Doc2Vec vectors.

    Args:
        movie_title (str): The title of the movie to get recommendations for.
        df (pd.DataFrame): DataFrame containing movie titles and their assigned 'cluster'.
                           It must also contain 'combined_features' for vector inference.
        d2v_model (Doc2Vec): The trained Doc2Vec model.
        top_n (int): The number of similar movies to return.
        similarity_cutoff (float): Cutoff for get_close_matches.

    Returns:
        Optional[List[str]]: A list of top_n similar movie titles,
                             or None if no close match is found or no recommendations.
    """
    # Find the closest match for the input movie title
    closest_matches = get_close_matches(movie_title, df['title'], n=1, cutoff=similarity_cutoff)
    
    if not closest_matches:
        print(f"No close match found for '{movie_title}'. Try a different title or adjust cutoff.")
        return None
    
    closest_match_title = closest_matches[0]
    print(f"Found closest match: '{closest_match_title}'")

    # Get the cluster of the matched movie
    try:
        movie_cluster = df[df['title'] == closest_match_title]['cluster'].values[0]
    except IndexError:
        print(f"Error: Could not find cluster for '{closest_match_title}'. Ensure 'cluster' column is present.")
        return None

    # Filter movies belonging to the same cluster
    cluster_movies_df = df[df['cluster'] == movie_cluster].copy() # Use .copy() to avoid SettingWithCopyWarning

    if cluster_movies_df.empty:
        print(f"No other movies found in the same cluster as '{closest_match_title}'.")
        return None

    # Infer vector for the input movie (closest match)
    input_movie_vector = d2v_model.infer_vector(
        df[df['title'] == closest_match_title]['combined_features'].iloc[0].split()
    ).reshape(1, -1) # Reshape for cosine_similarity

    # Infer vectors for all movies in the cluster
    cluster_movie_vectors = np.array(
        [d2v_model.infer_vector(text.split()) for text in cluster_movies_df['combined_features']]
    )

    # Calculate cosine similarity between the input movie and all movies in its cluster
    similarity_scores = cosine_similarity(input_movie_vector, cluster_movie_vectors).flatten()
    
    # Add similarity scores to the cluster DataFrame for sorting
    cluster_movies_df['similarity_score'] = similarity_scores
    
    # Sort movies by similarity (descending)
    similar_movies_df = cluster_movies_df.sort_values(by='similarity_score', ascending=False)
    
    # Exclude the input movie itself from the recommendations
    # and get the top_n movies
    recommended_movies = similar_movies_df[similar_movies_df['title'] != closest_match_title]['title'].head(top_n).tolist()

    if not recommended_movies:
        print(f"Found other movies in the cluster, but no new recommendations for '{closest_match_title}'.")
        return None
        
    return recommended_movies

# Main execution block
if __name__ == '__main__':
    # Configuration
    CSV_FILE_PATH = "titles.csv"
    D2V_VECTOR_SIZE = 100
    D2V_WINDOW = 5
    D2V_MIN_COUNT = 2 # Adjusted min_count
    D2V_WORKERS = 4
    D2V_EPOCHS = 20 # Number of epochs for Doc2Vec training
    SVD_N_COMPONENTS = 50 # Number of components for TruncatedSVD
    KMEANS_N_CLUSTERS = 10 # Number of clusters for K-Means
    RECOMMENDATIONS_TOP_N = 5

    # 1. Load data
    print("Loading data...")
    netflix_df_raw = load_data(CSV_FILE_PATH)

    # 2. Preprocess data
    print("Preprocessing data...")
    netflix_df_processed = preprocess_data(netflix_df_raw)

    # 3. Train Doc2Vec model
    print("Training Doc2Vec model...")
    doc2vec_model = train_doc2vec_model(
        netflix_df_processed,
        column_name='combined_features',
        vector_size=D2V_VECTOR_SIZE,
        window=D2V_WINDOW,
        min_count=D2V_MIN_COUNT,
        workers=D2V_WORKERS,
        epochs=D2V_EPOCHS
    )

    # 4. Apply dimensionality reduction
    print("Applying dimensionality reduction (SVD)...")
    reduced_feature_vectors = apply_dimensionality_reduction(
        doc2vec_model,
        netflix_df_processed,
        column_name='combined_features',
        n_components=SVD_N_COMPONENTS
    )

    # 5. Perform clustering
    print("Performing K-Means clustering...")
    kmeans_model, cluster_labels = perform_clustering(
        reduced_feature_vectors,
        n_clusters=KMEANS_N_CLUSTERS
    )
    
    # Add cluster labels to the DataFrame
    netflix_df_processed['cluster'] = cluster_labels
    print(f"Movies have been assigned to {KMEANS_N_CLUSTERS} clusters.")

    # 6. Get movie recommendations
    try:
        user_movie_title = input("Enter the movie title for recommendations: ")
        
        recommended_titles = get_recommendations(
            movie_title=user_movie_title,
            df=netflix_df_processed, # Pass the DataFrame with 'title', 'combined_features', and 'cluster'
            d2v_model=doc2vec_model,
            top_n=RECOMMENDATIONS_TOP_N
        )

        if recommended_titles:
            print(f"\nTop {RECOMMENDATIONS_TOP_N} movie recommendations for '{user_movie_title}':")
            for i, title in enumerate(recommended_titles):
                print(f"{i+1}. {title}")
        else:
            print(f"Sorry, could not find recommendations for '{user_movie_title}'.")

    except Exception as e:
        print(f"An error occurred during recommendation: {e}")
```
