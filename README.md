# Netflix Recommender Systems Using Cosine Similarity, Collaborative Filtering, and Matrix Factorization
 This project implements various approaches for building recommender systems, including Cosine Similarity, Collaborative Filtering (user-based and item-based), and Matrix Factorization using Alternating Least Squares (ALS). The system predicts ratings for users based on their movie preferences from the Netflix-like dataset.

# Building Recommender Systems Using Cosine Similarity, Collaborative Filtering, and Matrix Factorization

This project focuses on implementing different approaches for building recommender systems. It uses a user-item ratings dataset (e.g., Netflix) to compute cosine similarities, apply collaborative filtering algorithms, and implement matrix factorization using **Alternating Least Squares (ALS)**.

## Overview

The main goal of the project is to create a recommendation system for predicting ratings of movies based on historical user ratings. The project implements:

- **Cosine Similarity**: Calculates similarities between users or items to help make recommendations.
- **Collaborative Filtering**: Uses similarity scores to predict ratings based on neighbors' preferences.
- **Matrix Factorization (ALS)**: Decomposes the user-item matrix into lower-dimensional user and item feature matrices for more efficient predictions.

### Key Tasks:
1. **Cosine Similarity**: Compute the cosine similarity between users or items.
2. **Collaborative Filtering**:
    - **User-based CF**: Predict a user's rating for a movie based on similar users.
    - **Item-based CF**: Predict a user's rating for a movie based on similar items.
3. **Matrix Factorization (ALS)**: Apply ALS to decompose the matrix and predict ratings using latent factors.

### Dataset:
The dataset used contains user ratings for movies (e.g., Netflix-like data) in the following format:
- **UserID**: The ID of the user.
- **MovieID**: The ID of the movie.
- **Rating**: The rating given by the user to the movie.
- **Timestamp**: The timestamp of when the rating was given.

### Approach:
- **Cosine Similarity**: Compute the cosine similarity between vectors representing users or items, manually implementing the similarity formula.
- **Collaborative Filtering**: Implement both user-based and item-based collaborative filtering using the similarity matrix to predict ratings.
- **Matrix Factorization**: Use **Alternating Least Squares (ALS)** to iteratively update user and item matrices until convergence, to predict missing ratings.

### Algorithm Choices:
- **Cosine Similarity**: Manually computed using the formula:
  \[
  \text{cosine similarity}(A, B) = \frac{\sum_{i=1}^{n} a_i \cdot b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \cdot \sqrt{\sum_{i=1}^{n} b_i^2}}
  \]
- **Collaborative Filtering**: Implemented both **user-based** and **item-based** CF using the similarity matrix to predict ratings.
- **Matrix Factorization**: ALS is used to iteratively update user and item matrices until convergence, to predict missing ratings.

## Code Implementation

The code is structured as follows:

1. **`similarity_matrix`**: Computes cosine similarities between users or items.
2. **`user_based_cf`**: Implements user-based collaborative filtering to predict ratings.
3. **`item_based_cf`**: Implements item-based collaborative filtering to predict ratings.
4. **`UVDecomposition`**: A class implementing **Alternating Least Squares (ALS)** matrix factorization.

### Example Command:

```bash
python main.py -d /path/to/data.npy -s 2023 -m js

## Output

The results are stored in files like `js.txt`, `cs.txt`, or `dcs.txt` based on the selected similarity measure. Each file contains pairs of users that satisfy the corresponding similarity threshold.

For example:
- `js.txt`: Pairs of users with Jaccard similarity greater than 0.5.
- `cs.txt`: Pairs of users with Cosine similarity greater than 0.73.
- `dcs.txt`: Pairs of users with Discrete Cosine similarity greater than 0.73.

The output format for each file will be as follows:


## Results & Discussion

The project successfully implements the following methods:
- **Cosine Similarity**: Manually computed the similarity between users or items based on their ratings.
- **Collaborative Filtering**: Predictions were made for missing ratings based on the similarities between users or items.
- **Matrix Factorization (ALS)**: Used ALS to decompose the user-item matrix into latent features and predict ratings.

### Evaluation Metrics:
- The accuracy of the predictions is evaluated by comparing the predicted ratings with the actual ratings.
- **Cosine Similarity** provided a good approximation for user similarity, allowing for effective recommendations.
- **Collaborative Filtering** methods, both user-based and item-based, performed well but had limitations when dealing with sparse data.
- **Matrix Factorization (ALS)** showed improvement over traditional collaborative filtering, as it can handle sparse data better and provides latent feature representations for users and items.

### Performance:
- The implementation performed efficiently for the given dataset, and the time complexity was optimized by using matrix factorization and collaborative filtering techniques.
- **ALS** showed the best results in terms of prediction accuracy compared to the **user-based CF** and **item-based CF**, particularly when dealing with a larger matrix of users and movies.

## Conclusion

This project demonstrates different methods for building a recommender system, from traditional **collaborative filtering** to more sophisticated **matrix factorization** techniques. The results show that ALS matrix factorization can provide high-quality recommendations, while traditional CF methods such as user-based and item-based approaches remain effective for smaller datasets.

The key takeaway is that **matrix factorization** techniques like ALS are better suited for large datasets with a high degree of sparsity, as they allow us to factorize the matrix and uncover latent features that influence user preferences. In contrast, **collaborative filtering** methods provide solid results but struggle with scalability as the dataset grows.

## References

1. **Ricci, F., Rokach, L., & Shapira, B. (2015).** Introduction to Recommender Systems Handbook.
2. **Harper, F. M., & Konstan, J. A. (2015).** The MovieLens Datasets: History and Context.
3. **Koren, Y., Bell, R., & Volinsky, C. (2009).** Matrix Factorization Techniques for Recommender Systems. Computer Science Department, University of California, Berkeley.
