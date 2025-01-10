import pandas as pd
import numpy as np
import multiprocessing
import random

# Loading the dataset and pivoting it to create a user-item matrix
path = "u.data"
df = pd.read_table(path, sep="\t", names=["UserID", "MovieID", "Rating", "Timestamp"])
df = df.pivot_table(index='UserID', columns='MovieID', values='Rating')

# 1. COSINE SIMILARITY
def cosine_similarity_manual(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_vec1 = np.sqrt(sum(a * a for a in vec1))
    norm_vec2 = np.sqrt(sum(b * b for b in vec2))
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 and norm_vec2 else 0


def calculate_similarities(args):
    i, matrix, n, top_k = args
    similarities = []
    for j in range(n):
        if i != j:
            similarity = cosine_similarity_manual(matrix[i], matrix[j])
            similarities.append((j, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return i, [item[0] for item in similarities[:top_k]]

def similarity_matrix(user_item_matrix, axis=0, top_k=5):
    if axis == 0:
        matrix = user_item_matrix.fillna(0).values
    else:
        matrix = user_item_matrix.T.fillna(0).values

    n = matrix.shape[0]


    with multiprocessing.Pool() as pool:
        results = pool.map(calculate_similarities, [(i, matrix, n, top_k) for i in range(n)])

    similarity_dict = {i: neighbors for i, neighbors in results}
    return similarity_dict

# 2. COLLABORATIVE FILTERING
def user_based_cf(user_id, movie_id, user_similarity_matrix, user_item_matrix):
    neighbors = user_similarity_matrix.get(user_id - 1, [])
    numerator = sum(user_item_matrix.iloc[neighbor, movie_id - 1] * cosine_similarity_manual(user_item_matrix.iloc[user_id - 1], user_item_matrix.iloc[neighbor])
                    for neighbor in neighbors if not np.isnan(user_item_matrix.iloc[neighbor, movie_id - 1]))
    denominator = sum(abs(cosine_similarity_manual(user_item_matrix.iloc[user_id - 1], user_item_matrix.iloc[neighbor]))
                      for neighbor in neighbors if not np.isnan(user_item_matrix.iloc[neighbor, movie_id - 1]))
    return numerator / denominator if denominator != 0 else 0

def item_based_cf(user_id, movie_id, item_similarity_matrix, user_item_matrix):
    neighbors = item_similarity_matrix.get(movie_id - 1, [])
    numerator = sum(user_item_matrix.iloc[user_id - 1, neighbor] * cosine_similarity_manual(user_item_matrix.iloc[:, neighbor], user_item_matrix.iloc[:, movie_id - 1])
                    for neighbor in neighbors if not np.isnan(user_item_matrix.iloc[user_id - 1, neighbor]))
    denominator = sum(abs(cosine_similarity_manual(user_item_matrix.iloc[:, neighbor], user_item_matrix.iloc[:, movie_id - 1]))
                      for neighbor in neighbors if not np.isnan(user_item_matrix.iloc[user_id - 1, neighbor]))
    return numerator / denominator if denominator != 0 else 0

# 3. MATRIX FACTORIZATION USING ALS
class UVDecomposition:
    def __init__(self, user_item_matrix, num_features=10, regularization=0.1, iterations=10):
        self.user_item_matrix = user_item_matrix
        self.num_users, self.num_items = user_item_matrix.shape
        self.num_features = num_features
        self.regularization = regularization
        self.iterations = iterations

        self.U = np.random.rand(self.num_users, self.num_features)
        self.V = np.random.rand(self.num_items, self.num_features)

    def update_U(self):
        for i in range(self.num_users):
            VtV = self.V.T.dot(self.V) + self.regularization * np.eye(self.num_features)
            VtR = self.V.T.dot(self.user_item_matrix.iloc[i, :].fillna(0).values)
            self.U[i, :] = np.linalg.solve(VtV, VtR)

    def update_V(self):
        for j in range(self.num_items):
            UtU = self.U.T.dot(self.U) + self.regularization * np.eye(self.num_features)
            UtR = self.U.T.dot(self.user_item_matrix.iloc[:, j].fillna(0).values)
            self.V[j, :] = np.linalg.solve(UtU, UtR)

    def train(self):
        for iteration in range(self.iterations):
            self.update_U()
            self.update_V()
        return self.U, self.V

    def predict(self, user, item):
        user_index = user - 1
        item_index = item - 1
        prediction = np.dot(self.U[user_index, :], self.V[item_index, :].T)
        return prediction

if __name__ == "__main__":
    user_similarity_matrix = similarity_matrix(df, top_k=5, axis=0)
    print(user_similarity_matrix.get(2, []))

    item_similarity_matrix = similarity_matrix(df, top_k=5, axis=1)
    print(item_similarity_matrix.get(9, []))

    user_id = 13
    movie_id = 100

    u_predicted_rating = user_based_cf(user_id, movie_id, user_similarity_matrix, user_item_matrix=df)
    print(f"Predicted user {user_id} rating for movie {movie_id} using user-based CF: {u_predicted_rating:.2f}")

    i_predicted_rating = item_based_cf(user_id, movie_id, item_similarity_matrix, user_item_matrix=df)
    print(f"Predicted user {user_id} rating for movie {movie_id} using item-based CF: {i_predicted_rating:.2f}")

    uv_model = UVDecomposition(df, num_features=10, iterations=5)
    uv_model.train()
    predicted_rating = uv_model.predict(user_id, movie_id)
    print(f"Predicted rating by UV-decomposition method: {predicted_rating:.2f}")
