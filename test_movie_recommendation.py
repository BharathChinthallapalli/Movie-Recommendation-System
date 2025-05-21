import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock # Using MagicMock for more flexibility if needed later

# Assuming get_recommendations is in MovieRecommendationSystem.py
# Ensure MovieRecommendationSystem.py is in the PYTHONPATH or same directory
from MovieRecommendationSystem import get_recommendations

# Helper Mock Class for Doc2Vec
class MockDoc2Vec:
    """
    A mock Doc2Vec model that returns predefined vectors for specific inputs.
    """
    def __init__(self):
        # Store vectors as a dictionary where keys are tuple(words)
        self.vectors = {
            ('feature', 'A'): np.array([1.0, 0.8, 0.2]), # Base vector for Movie A
            ('feature', 'B'): np.array([0.9, 0.7, 0.3]), # Similar to A
            ('feature', 'C'): np.array([-0.5, 0.5, -0.5]),# Different from A and B (different cluster)
            ('feature', 'D'): np.array([0.8, 0.9, 0.1]), # Similar to A, but maybe less so than B
            ('feature', 'E'): np.array([1.0, 0.8, 0.2]), # Identical to A (for exact match test)
        }
        self.default_vector = np.array([0.0, 0.0, 0.0]) # For unknown words

    def infer_vector(self, words: list[str]) -> np.ndarray:
        # Convert list of words to tuple to use as dict key
        key = tuple(words)
        return self.vectors.get(key, self.default_vector)

class TestMovieRecommendationSystem(unittest.TestCase):

    def setUp(self):
        """
        Set up mock data and models for each test method.
        """
        self.mock_data = {
            'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
            'combined_features': [
                'feature A', # Corresponds to ('feature', 'A')
                'feature B', # Corresponds to ('feature', 'B')
                'feature C', # Corresponds to ('feature', 'C')
                'feature D', # Corresponds to ('feature', 'D')
                'feature E'  # Corresponds to ('feature', 'E')
            ],
            'cluster': [0, 0, 1, 0, 0]  # A, B, D, E in cluster 0; C in cluster 1
        }
        self.mock_df = pd.DataFrame(self.mock_data)
        self.mock_d2v_model = MockDoc2Vec()

    def test_get_recommendations_smoke_test(self):
        """
        Test basic recommendation functionality:
        - Movie A is input.
        - Movie B and Movie D are in the same cluster (0) and should be candidates.
        - Movie E is identical to Movie A and should be excluded.
        - Movie C is in a different cluster (1) and should not be recommended.
        - Expect Movie B or Movie D based on similarity (mocked vectors are close).
        """
        recommendations = get_recommendations(
            movie_title="Movie A",
            df=self.mock_df,
            d2v_model=self.mock_d2v_model,
            top_n=1,
            similarity_cutoff=0.6 # Default from main script refactor
        )
        self.assertIsNotNone(recommendations, "Recommendations should not be None.")
        self.assertIsInstance(recommendations, list, "Recommendations should be a list.")
        self.assertEqual(len(recommendations), 1, "Should recommend 1 movie.")
        # Based on our mock vectors, Movie D is slightly more similar to A than B if we only consider the first component,
        # but cosine similarity will look at all.
        # infer_vector for 'feature A' is [1.0, 0.8, 0.2]
        # infer_vector for 'feature B' is [0.9, 0.7, 0.3] -> cos_sim with A ~0.99
        # infer_vector for 'feature D' is [0.8, 0.9, 0.1] -> cos_sim with A ~0.97
        # So Movie B should be recommended over Movie D if top_n=1
        self.assertIn("Movie B", recommendations, "Movie B should be recommended for Movie A.")

    def test_get_recommendations_top_2(self):
        """
        Test with top_n=2 to get more recommendations from the same cluster.
        """
        recommendations = get_recommendations(
            movie_title="Movie A",
            df=self.mock_df,
            d2v_model=self.mock_d2v_model,
            top_n=2,
            similarity_cutoff=0.6
        )
        self.assertIsNotNone(recommendations)
        self.assertIsInstance(recommendations, list)
        self.assertEqual(len(recommendations), 2)
        self.assertIn("Movie B", recommendations)
        self.assertIn("Movie D", recommendations)
        # Check order based on similarity (B then D)
        self.assertEqual(recommendations[0], "Movie B")
        self.assertEqual(recommendations[1], "Movie D")


    def test_get_recommendations_no_match_for_title(self):
        """
        Test behavior when the input movie title has no close matches.
        """
        recommendations = get_recommendations(
            movie_title="NonExistent Movie ZXC", # A title very unlikely to match
            df=self.mock_df,
            d2v_model=self.mock_d2v_model,
            top_n=1,
            similarity_cutoff=0.8 # Using a slightly higher cutoff as in original script
        )
        self.assertIsNone(recommendations, "Should return None for no close title match.")

    def test_get_recommendations_different_cluster(self):
        """
        Test that movies from different clusters are not recommended,
        even if their features might be somewhat similar.
        Movie C is in cluster 1. Movie A is in cluster 0.
        """
        # Temporarily make 'feature C' vector very similar to 'feature A'
        # to ensure clustering is the primary filter here.
        original_vector_c = self.mock_d2v_model.vectors[('feature', 'C')]
        self.mock_d2v_model.vectors[('feature', 'C')] = np.array([0.99, 0.79, 0.19]) # Very similar to A

        recommendations = get_recommendations(
            movie_title="Movie A",
            df=self.mock_df,
            d2v_model=self.mock_d2v_model,
            top_n=1
        )
        self.assertIsNotNone(recommendations)
        self.assertNotIn("Movie C", recommendations, "Movie C from a different cluster should not be recommended.")
        
        # Restore original vector for other tests
        self.mock_d2v_model.vectors[('feature', 'C')] = original_vector_c

    def test_get_recommendations_input_movie_excluded(self):
        """
        Test that the input movie itself is not included in the recommendations.
        Movie E has identical features to Movie A and is in the same cluster.
        """
        recommendations = get_recommendations(
            movie_title="Movie A", # Input movie
            df=self.mock_df,
            d2v_model=self.mock_d2v_model,
            top_n=3 # Request more movies than available similar ones (excluding input)
        )
        self.assertIsNotNone(recommendations)
        self.assertNotIn("Movie A", recommendations, "Input movie 'Movie A' should not be in recommendations.")
        self.assertIn("Movie B", recommendations)
        self.assertIn("Movie D", recommendations)
        self.assertIn("Movie E", recommendations) # Movie E is similar enough to A and in same cluster
        self.assertEqual(len(recommendations), 3)


    def test_get_recommendations_no_other_movies_in_cluster(self):
        """
        Test behavior when the matched movie is the only one in its cluster.
        """
        single_movie_data = {
            'title': ['Lonely Movie'],
            'combined_features': ['feature lonely'],
            'cluster': [0]
        }
        single_movie_df = pd.DataFrame(single_movie_data)
        
        # Ensure mock d2v model knows about 'feature lonely'
        self.mock_d2v_model.vectors[('feature', 'lonely')] = np.array([0.1, 0.2, 0.3])

        recommendations = get_recommendations(
            movie_title="Lonely Movie",
            df=single_movie_df,
            d2v_model=self.mock_d2v_model,
            top_n=1
        )
        # The current implementation returns None if no *other* movies are found for recommendation.
        # This depends on the exact logic: "Found other movies in the cluster, but no new recommendations"
        # or "No other movies found in the same cluster".
        # Given the current implementation, it prints "No other movies found..." and returns None.
        # If it were an empty list, the assertion would be `self.assertEqual(recommendations, [])`
        self.assertIsNone(recommendations, "Should return None if no other movies are in the cluster.")

    def test_get_recommendations_cutoff_impact(self):
        """
        Test how similarity_cutoff for get_close_matches affects results.
        """
        # First, a successful match with default cutoff
        recommendations_low_cutoff = get_recommendations(
            movie_title="Movie A", # Exact match
            df=self.mock_df,
            d2v_model=self.mock_d2v_model,
            top_n=1,
            similarity_cutoff=0.6 # Lower cutoff ensures "Movie A" is found
        )
        self.assertIsNotNone(recommendations_low_cutoff)
        self.assertIn("Movie B", recommendations_low_cutoff)

        # Now, a very high cutoff that should prevent "Movie A" from being matched
        recommendations_high_cutoff = get_recommendations(
            movie_title="Movie A", # Title exists
            df=self.mock_df,
            d2v_model=self.mock_d2v_model,
            top_n=1,
            similarity_cutoff=0.99 # difflib scores are 0-1, 1 is perfect.
                                    # "Movie A" to "Movie A" is 1.0.
                                    # If we search for "Movie A" it will be a perfect match.
                                    # Let's try with a slight variation that doesn't exist to test the cutoff
        )
        # This test is a bit tricky with get_close_matches. If an exact match occurs, cutoff is less relevant.
        # Let's test a non-existent but *close* title
        recommendations_close_title_high_cutoff = get_recommendations(
            movie_title="Movie AA", # Close to "Movie A"
            df=self.mock_df,
            d2v_model=self.mock_d2v_model,
            top_n=1,
            similarity_cutoff=0.95 # "Movie AA" vs "Movie A" might be ~0.88, so 0.95 should fail
        )
        self.assertIsNone(recommendations_close_title_high_cutoff, "High cutoff should prevent matching 'Movie AA' to 'Movie A'.")

        recommendations_close_title_lower_cutoff = get_recommendations(
            movie_title="Movie AA", # Close to "Movie A"
            df=self.mock_df,
            d2v_model=self.mock_d2v_model,
            top_n=1,
            similarity_cutoff=0.8 # "Movie AA" vs "Movie A" might be ~0.88, so 0.8 should find it
        )
        self.assertIsNotNone(recommendations_close_title_lower_cutoff, "Lower cutoff should allow matching 'Movie AA' to 'Movie A'.")
        if recommendations_close_title_lower_cutoff: # Guard for safety
            self.assertIn("Movie B", recommendations_close_title_lower_cutoff) # Assuming Movie A is matched, then B is recommended

if __name__ == '__main__':
    unittest.main()
```
