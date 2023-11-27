from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Cosine_similarity:
    def calculate_cosine_similarity(self, vec1, vec2):
        return cosine_similarity(np.array([vec1]), np.array([vec2]))[0, 0]

    def find_most_similar_movies(self, user_input_vector, movie_vectors, num_similar=5):
        similarity_scores = {}
        for movie_id, movie_vector in movie_vectors:
            similarity = self.calculate_cosine_similarity(
                user_input_vector, movie_vector
            )
            if movie_id in similarity_scores:
                similarity_scores[movie_id] = max(
                    similarity_scores[movie_id], similarity
                )
            else:
                similarity_scores[movie_id] = similarity
        most_similar_movies = sorted(
            similarity_scores.items(), key=lambda x: x[1], reverse=True
        )[:num_similar]
        return most_similar_movies
