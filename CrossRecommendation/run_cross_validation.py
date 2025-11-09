'''
README!

To compare performance with our model using the pre-trained LightFM model as baseline.

In the final recommendation, our model gives a high score and only recommends a list of recommendations similar to LightFM's results
'''

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# --- 0. Loading Modules and Models ---
print("--- Start loading models and data ---")

# Recommender class import for loading 'Our Model'
from recommender import CrossDomainRecommender

# Importing libraries for LightFM loading
from lightfm import LightFM

# --- 1. Setting the Path ---
# File path that our model will use
PROFILE_PATH = './processed_data/profile/'
BOOK_GENRE_PROFILE_PATH = PROFILE_PATH + 'book_user_genre_profile_ensemble.csv'
MOVIE_GENRE_PROFILE_PATH = PROFILE_PATH + 'movie_user_genre_profile_ensemble.csv'
BOOK_DECADE_PROFILE_PATH = PROFILE_PATH + 'book_user_decade_profile.csv'
MOVIE_DECADE_PROFILE_PATH = PROFILE_PATH + 'movie_user_decade_profile.csv'

SPLIT_DATA_PATH = './processed_data/splitData/'
BOOK_TRAIN_PATH = SPLIT_DATA_PATH + 'book_train.csv'
MOVIE_TRAIN_PATH = SPLIT_DATA_PATH + 'movielens_train.csv'
BOOK_TEST_PATH = SPLIT_DATA_PATH + 'book_test.csv'

# LightFM model path
LIGHTFM_FOLDER = './processed_data/lightfm/'
LIGHTFM_MODEL_PATH = LIGHTFM_FOLDER + 'lightfm_model.pkl'
LIGHTFM_DATASET_PATH = LIGHTFM_FOLDER + 'lightfm_dataset.pkl'

# --- 2. Model Initialization ---
try:
    # 2-1. Initialize 'Our Model' (Transfer Learning Data Path)
    our_recommender = CrossDomainRecommender(
        book_genre_profile_path=BOOK_GENRE_PROFILE_PATH,
        movie_genre_profile_path=MOVIE_GENRE_PROFILE_PATH,
        book_decade_profile_path=BOOK_DECADE_PROFILE_PATH,
        movie_decade_profile_path=MOVIE_DECADE_PROFILE_PATH,
        book_data_path=BOOK_TRAIN_PATH,
        movie_data_path=MOVIE_TRAIN_PATH
    )

    # 2-2. Load 'LightFM' Model and Dataset
    with open(LIGHTFM_MODEL_PATH, 'rb') as f: lightfm_model = pickle.load(f)
    with open(LIGHTFM_DATASET_PATH, 'rb') as f: lightfm_dataset = pickle.load(f)
    user_mapping, _, item_mapping, _ = lightfm_dataset.mapping()
    
    print("-> All models loaded.")

except FileNotFoundError as e:
    print(f"[Fatal error] File not found for model loading: {e}")
    print("-> Please check if you ran 'run_preprocessing.py' and 'lightfm_baseline.py' first.")
    exit()

# --- 3. LightFM latent vector extraction function ---
def get_lightfm_vectors(model, dataset_mapping, ids, id_prefix):
    """Extract the latent vectors (embedding) of specific IDs from the LightFM model."""
    # Use only pure latent vectors except item biases
    _, item_embeddings = model.get_item_representations(features=None)
    
    # Converting the input IDs into LightFM internal indexes
    indices = [dataset_mapping.get(id_prefix + str(i)) for i in ids if (id_prefix + str(i)) in dataset_mapping]
    if not indices: return None
    
    return item_embeddings[indices]

# --- 4. Evaluation Loop ---
def main():
    print("\n--- Start cross-model validation assessments ---")
    
    # Loading Test User ID List
    test_books_df = pd.read_csv(BOOK_TEST_PATH)
    test_user_ids = test_books_df['user_id'].unique()
    
    average_similarities = []

    # Only 200 people run first for testing
    for user_id in tqdm(test_user_ids[:200], desc="Validating Recommendations"):
        # a. Get movie recommendations as our model
        recommendations, _ = our_recommender.recommend_movies(user_id, top_n=10)
        if recommendations is None or recommendations.empty: continue
        recommended_movie_ids = recommendations.index.tolist()

        # b. Get a list of books that this user likes from the test set
        user_liked_books = test_books_df[
            (test_books_df['user_id'] == user_id) & (test_books_df['rating'] >= 4)
        ]['book_id'].tolist()
        if not user_liked_books: continue

        # c. Extract vectors from LightFM space
        # Average Vector of the user's favorite books ("Real" book taste vector)
        liked_books_vectors = get_lightfm_vectors(lightfm_model, item_mapping, user_liked_books, 'book_')
        if liked_books_vectors is None: continue
        actual_taste_vector = liked_books_vectors.mean(axis=0).reshape(1, -1)

        # Vectors of recommended movies
        recommended_movies_vectors = get_lightfm_vectors(lightfm_model, item_mapping, recommended_movie_ids, 'movie_')
        if recommended_movies_vectors is None: continue

        # d. Calculation of similarity
        sims = cosine_similarity(actual_taste_vector, recommended_movies_vectors)[0]
        
        # e. Save the average similarity for this user
        average_similarities.append(np.mean(sims))

    # --- 5. The final result ---
    final_avg_similarity = np.mean(average_similarities) if average_similarities else 0
    print("\n\n--- Final evaluation results ---")
    print(f"Average LightFM spatial similarity between our model's recommendation results and user book tastes: {final_avg_similarity:.4f}")

if __name__ == '__main__':
    main()