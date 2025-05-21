# Movie Recommendation System

This Python-based movie recommendation system helps you discover new movie favorites. It uses data processing, feature engineering, and machine learning techniques to provide personalized recommendations.

## Methodology

The system follows this pipeline to generate recommendations:

1.  **Data Loading and Preprocessing:** Loads movie data from `titles.csv`. It then preprocesses the data by combining textual features (`title`, `type`, `production_countries`, `description`) into a single string for each movie. This consolidated text forms the basis for understanding the movie's content.
2.  **Doc2Vec Embeddings:** Uses Gensim's Doc2Vec to learn dense vector representations (embeddings) for each movie based on its combined textual features. These embeddings capture semantic similarities between movies, meaning movies with similar plots, genres, or themes will have similar vector representations.
3.  **Dimensionality Reduction:** Applies TruncatedSVD (Singular Value Decomposition) to reduce the dimensionality of the Doc2Vec embeddings. This step helps to remove noise, reduce computational complexity, and can improve the performance of the subsequent clustering step.
4.  **K-Means Clustering:** Groups movies into a predefined number of clusters using the K-Means algorithm. Clustering is performed on the reduced embeddings, so movies within the same cluster are considered broadly similar in terms of their content and themes.
5.  **Recommendation:** When a user provides a movie title, the system first tries to find the closest match in its database. It then identifies the cluster to which this movie belongs. Finally, it recommends other movies from the same cluster based on the cosine similarity of their original Doc2Vec vectors. Movies with higher cosine similarity scores to the input movie are considered more similar and are recommended.

## Dataset

The system requires a CSV file named `titles.csv` to be present in the same directory as the script (`MovieRecommendationSystem.py`).

The `titles.csv` file must contain the following key columns for the system to function correctly:
*   `title`: The title of the movie.
*   `type`: The type of content (e.g., Movie, TV Show).
*   `production_countries`: A list or string representing the production countries.
*   `description`: A textual description or synopsis of the movie.

[Note: You will need to provide your own `titles.csv` file. Ensure it contains the aforementioned columns. Publicly available movie datasets from platforms like Kaggle (e.g., Netflix Prize dataset, MovieLens datasets) can be adapted for this purpose. You might need to preprocess these datasets to match the required column names and formats.]

## Setup and Usage

Follow these steps to set up and run the movie recommendation system:

1.  **Create and activate a virtual environment (Optional but recommended):**
    ```bash
    python -m venv venv
    # On macOS/Linux
    source venv/bin/activate
    # On Windows
    # venv\Scripts\activate
    ```
2.  **Install dependencies:**
    Ensure you have the `requirements.txt` file from this repository. Then run:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Prepare the dataset:**
    Place your `titles.csv` file (as described in the "Dataset" section) in the same directory as the `MovieRecommendationSystem.py` script.
4.  **Run the script:**
    ```bash
    python MovieRecommendationSystem.py
    ```
    The script will prompt you to "Enter the movie title: ". Type a movie title you like and press Enter to get recommendations.

## Future Improvements

This project provides a solid foundation for a content-based movie recommendation system. Here are some potential enhancements:

*   **Evaluating Recommendation Quality:** While this project sets up a functional recommendation pipeline, a crucial next step in a real-world scenario would be to rigorously evaluate the quality of recommendations. The current "Model Evaluation" point can be expanded as follows:
    *   **Qualitative Evaluation:**
        *   **User Feedback:** Collect feedback from users on how relevant and satisfying they find the recommendations. This can be done through surveys or user studies.
        *   **Case Studies:** Manually examine recommendations for a diverse set of input movies to assess their coherence and usefulness.
    *   **Quantitative Evaluation (Offline Metrics):**
        *   If historical user interaction data were available (e.g., movies users have watched or rated highly), we could use metrics like:
            *   **Precision@k:** What proportion of the top K recommended movies are actually relevant to the user?
            *   **Recall@k:** Of all the movies relevant to a user, what proportion did we recommend in the top K?
            *   **nDCG@k (Normalized Discounted Cumulative Gain):** A measure of ranking quality that gives higher scores for relevant items appearing earlier in the recommendation list.
        *   These metrics require a dataset with known positive interactions (ground truth). For a content-based system like this, one could also define "relevance" based on shared attributes (e.g., genre, keywords) if explicit user feedback isn't available, though this is less ideal than true user interaction data.
    *   **Online Evaluation (A/B Testing):**
        *   In a live system, different recommendation algorithms or parameters could be tested simultaneously with different user groups (A/B testing) to see which performs better based on metrics like click-through rate, conversion rate, or user engagement.
    *   **Beyond Accuracy - Other Important Aspects:**
        *   **Diversity:** How different are the recommended movies from each other? Highly similar recommendations might be less useful. Metrics like intra-list similarity can be used.
        *   **Serendipity:** Does the system recommend surprising yet relevant items that the user might not have discovered otherwise?
        *   **Coverage:** What percentage of the total movie catalog can be recommended? It's important to ensure the system doesn't only recommend popular items.
        *   **Novelty:** Does the system recommend items unknown to the user?

*   **Hyperparameter Tuning:** Systematically experiment with different parameters for Doc2Vec (e.g., vector size, window size, min_count, epochs), TruncatedSVD (number of components), and K-Means (number of clusters) to optimize the model's performance based on the evaluation metrics chosen above. Techniques like grid search or randomized search can be employed.
*   **Alternative Algorithms:** Explore other recommendation techniques:
    *   **Collaborative Filtering:** If user-item interaction data (e.g., ratings) is available, collaborative filtering methods (e.g., matrix factorization, k-NN) could be implemented.
    *   **Content-Based Filtering with TF-IDF:** Use TF-IDF vectors instead of or in conjunction with Doc2Vec embeddings.
    *   **Hybrid Models:** Combine content-based and collaborative filtering approaches to leverage the strengths of both.
*   **User Interface:** Develop a simple web application (e.g., using Flask or Streamlit) to provide a more interactive and user-friendly experience for getting recommendations.
*   **Scalability:** For significantly larger datasets, explore more scalable solutions for data processing (e.g., Spark), embedding generation, and model serving (e.g., using a dedicated API).
*   **Advanced Feature Engineering:** Incorporate other features like genres, actors, directors, or release year to enrich the movie representations.
*   **Cold Start Problem:** Investigate strategies to handle new movies or movies with sparse textual information.

This code is a valuable addition to any data science portfolio as it showcases skills in data processing, feature engineering, natural language processing, and machine learning. I'm proud of the results it produces and how it can enhance the movie-watching experience. Try it out to discover new movie favorites! Happy movie hunting!
