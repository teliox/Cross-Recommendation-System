# main.py

import pandas as pd
import numpy as np
import random
import pickle
import os

from recommender import CrossDomainRecommender
from lightfm import LightFM
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Setting path and hyperparameter ---
#Collects all settings at the top of the file to define them.
SPLIT_DATA_PATH = './processed_data/splitData/'
PROFILE_PATH = './processed_data/profile/'
LIGHTFM_FOLDER = './processed_data/lightfm/'

BOOK_TRAIN_PATH = SPLIT_DATA_PATH + 'book_train.csv'
MOVIE_TRAIN_PATH = SPLIT_DATA_PATH + 'movielens_train.csv'
BOOK_TEST_PATH = SPLIT_DATA_PATH + 'book_test.csv'
MOVIE_DATA_PATH = './processed_data/finalDataset/movielens_preprocessed_final.csv' # need full movie information

BOOK_GENRE_PROFILE_PATH = PROFILE_PATH + 'book_user_genre_profile_ensemble.csv'
MOVIE_GENRE_PROFILE_PATH = PROFILE_PATH + 'movie_user_genre_profile_ensemble.csv'
BOOK_DECADE_PROFILE_PATH = PROFILE_PATH + 'book_user_decade_profile.csv'
MOVIE_DECADE_PROFILE_PATH = PROFILE_PATH + 'movie_user_decade_profile.csv'

LIGHTFM_MODEL_PATH = LIGHTFM_FOLDER + 'lightfm_model.pkl'
LIGHTFM_DATASET_PATH = LIGHTFM_FOLDER + 'lightfm_dataset.pkl'

# Model hyperparameters
GENRE_WEIGHT = 0.5
SEMANTIC_WEIGHT = 0.5
K_NEIGHBORS = 20
TOP_N_RECOMMENDATIONS = 10

# --- 2. LightFM helper function ---
def get_lightfm_vectors(model, dataset_mapping, ids, id_prefix):
    """Extract the latent vectors (embedding) of specific IDs from the LightFM model."""
    _, item_embeddings = model.get_item_representations(features=None)
    indices = [dataset_mapping.get(id_prefix + str(i)) for i in ids if (id_prefix + str(i)) in dataset_mapping]
    if not indices: return None
    return item_embeddings[indices]

def recommend_with_lightfm(user_id, model, dataset, movie_info_df, top_n=10):
    """We recommend movies to book users with LightFM models."""
    user_mapping, _, item_mapping, _ = dataset.mapping()
    user_id_str = 'book_' + str(user_id)
    
    if user_id_str not in user_mapping:
        print(f"-> LightFM: User '{user_id}' is not in training data.")
        return None
        
    user_idx = user_mapping[user_id_str]
    _, user_embeddings = model.get_user_representations()
    target_user_vector = user_embeddings[user_idx]
    
    all_movie_ids_str = [item for item in item_mapping.keys() if item.startswith('movie_')]
    all_movie_indices = [item_mapping[mid] for mid in all_movie_ids_str]
    
    _, item_embeddings = model.get_item_representations()
    all_movie_vectors = item_embeddings[all_movie_indices]
    
    scores = cosine_similarity(target_user_vector.reshape(1, -1), all_movie_vectors)[0]
    
    top_indices_in_list = np.argsort(-scores)[:top_n]
    
    recommended_movie_ids_str = [all_movie_ids_str[i] for i in top_indices_in_list]
    recommended_scores = scores[top_indices_in_list]
    recommended_movie_ids = [int(mid.replace('movie_', '')) for mid in recommended_movie_ids_str]

    recommendations = pd.DataFrame({'movieId': recommended_movie_ids, 'predicted_score': recommended_scores})
    final_recs = pd.merge(recommendations, movie_info_df, on='movieId').set_index('movieId')
    return final_recs

# --- 3. Main Analytical Function ---
def analyze_single_user(user_id, our_recommender, lightfm_model, lightfm_dataset, book_test_df, movie_info_df):
    """
    Runs a complete analysis pipeline for a specific book user.
    1. Create and analyze 'Our Model' recommendations
    2. Create 'LightFM' baseline recommendations
    3. Mutual verification using 'LightFM' space
    4. Present a list of final recommendations that have passed the cross-validation
    """
    print(f"\n\n{'='*25} Start deep analysis of user '{user_id}' {'='*25}")

    # --- A. Create primary recommendation with 'Our Model' ---
    print("\n--- 1. 'Our Model' Recommendation generate ---")
    recommendations_df, neighbor_info = our_recommender.recommend_movies(
        book_user_id=user_id,
        k=K_NEIGHBORS,
        top_n=TOP_N_RECOMMENDATIONS
    )

    if recommendations_df is None or recommendations_df.empty:
        print(f"-> 'Our Model' failed to generate recommendation. Ends analysis.")
        print(f"{'='*80}")
        return

    # --- B. Output detailed analysis of 'Our Model' recommendation results ---
    print(f"\n--- 2. Analyzing 'Our Model' recommendation results ---")
    
    if 'user_top_genres' in recommendations_df.columns and not recommendations_df['user_top_genres'].empty:
        top_genres = recommendations_df['user_top_genres'].iloc[0]
        print(f"\n** Top 5 Preferred Genre of the user being analyzed **\n{top_genres}")
    
    if neighbor_info is not None:
        print("\n** About the Top 5 Similar Neighbors (Movie Users) selected **\n", neighbor_info.head(5))

    print("\n** List of Top 10 Recommended Movies (Our Model - 1st) **")
    print(recommendations_df.drop('user_top_genres', axis=1, errors='ignore'))

    # --- C. "LightFM" baseline recommendation results ---
    print("\n\n--- 3. 'LightFM' baseline recommendation results ---")
    lightfm_recs = recommend_with_lightfm(user_id, lightfm_model, lightfm_dataset, movie_info_df, top_n=TOP_N_RECOMMENDATIONS)
    if lightfm_recs is None:
        print("-> 'LightFM' failed to generate recommendation.")
    else:
        print("\n** List of Top 10 Recommended Movies (LightFM) **")
        print(lightfm_recs)

    # --- D. Verification of 'Our Model' results using 'LightFM' ---
    print(f"\n\n--- 4. 'Our Model' recommendation result verification (by LightFM) ---")
    
    user_liked_books = book_test_df[(book_test_df['user_id'] == user_id) & (book_test_df['rating'] >= 4)]['book_id'].tolist()
    
    if not user_liked_books:
        print("\n-> Validation failed: This user did not have any books he liked in the test set.")
        print(f"{'='*80}")
        return
        
    print(f"** User Favourite Books (Top 5 Books): {user_liked_books[:5]} **")

    liked_books_vectors = get_lightfm_vectors(lightfm_model, item_mapping, user_liked_books, 'book_')
    if liked_books_vectors is None:
        print("-> Validation failed: No LightFM vector found for your favorite book.")
        print(f"{'='*80}")
        return
        
    actual_taste_vector = liked_books_vectors.mean(axis=0).reshape(1, -1)
    recommended_movie_ids = recommendations_df.index.tolist()
    recommended_movies_vectors = get_lightfm_vectors(lightfm_model, item_mapping, recommended_movie_ids, 'movie_')
    
    if recommended_movies_vectors is None:
        print("-> Validation failed: No LightFM vector found for the recommended movie.")
        print(f"{'='*80}")
        return

    similarities = cosine_similarity(actual_taste_vector, recommended_movies_vectors)[0]
    
    validation_df = pd.DataFrame({
        'Similarity with Liked Books (in LightFM space)': similarities
    }, index=recommendations_df.index)
    
    # Combine original recommendation with verification score
    validation_df = validation_df.join(recommendations_df)
    validation_df = validation_df.sort_values('Similarity with Liked Books (in LightFM space)', ascending=False)

    print("\n** Verification Results: LightFM Space Similarity Between 'Favorite Books' and 'Recommended Movies' **")
    print(validation_df[['title_cleaned', 'Similarity with Liked Books (in LightFM space)']])
    
    final_avg_similarity = np.mean(similarities)
    print(f"\n>> Overall mean similarity score: {final_avg_similarity:.4f}")

    # --- E. Mutual Validation Based Final Recommendation ---
    print(f"\n\n--- 5. Mutual Validation Based Final Recommendation list ---")
    
    # Filter only films with positive similarity scores in verification results
    final_recommendations = validation_df[validation_df['Similarity with Liked Books (in LightFM space)'] > 0]
    
    if final_recommendations.empty:
        print("\n-> We did not find any recommended movies that the two models agreed on in common.")
    else:
        print("\n** Final recommended films that both models rated positively (in order of high similarity) **")
        # Select the column to show
        display_cols = ['title_cleaned', 'genres', 'final_score', 'Similarity with Liked Books (in LightFM space)']
        available_cols = [col for col in display_cols if col in final_recommendations.columns]
        print(final_recommendations[available_cols])
        
    print(f"{'='*80}")

if __name__ == '__main__':
    try:
        # --- Loading Model and Data ---
        
        ##############################################################
        # When the Recommender is initialized, it delivers the necessary weight parameters.
        our_recommender = CrossDomainRecommender(
            book_genre_profile_path=BOOK_GENRE_PROFILE_PATH,
            movie_genre_profile_path=MOVIE_GENRE_PROFILE_PATH,
            book_decade_profile_path=BOOK_DECADE_PROFILE_PATH,
            movie_decade_profile_path=MOVIE_DECADE_PROFILE_PATH,
            book_data_path=BOOK_TRAIN_PATH,
            movie_data_path=MOVIE_TRAIN_PATH,
            genre_weight=GENRE_WEIGHT, # Transfer this value
            semantic_weight=SEMANTIC_WEIGHT # Transfer this value
        )
        ##############################################################

        with open(LIGHTFM_MODEL_PATH, 'rb') as f: lightfm_model = pickle.load(f)
        with open(LIGHTFM_DATASET_PATH, 'rb') as f: lightfm_dataset = pickle.load(f)
        user_mapping, _, item_mapping, _ = lightfm_dataset.mapping()
        
        book_test_df = pd.read_csv(BOOK_TEST_PATH)
        #Movie_info_df is more stable to use the entire movie data.
        movie_info_df = pd.read_csv(MOVIE_DATA_PATH)[['movieId', 'title_cleaned', 'genres']].drop_duplicates()

        # --- Perform analysis---
        test_user_ids = book_test_df['user_id'].unique()
        
        # Perform analysis on 3 random users
        if len(test_user_ids) > 3:
            sample_user_ids = random.sample(list(test_user_ids), 3)
        else:
            sample_user_ids = test_user_ids # Use all if there are fewer than 3 users

        for user_id in sample_user_ids:
            analyze_single_user(user_id, our_recommender, lightfm_model, lightfm_dataset, book_test_df, movie_info_df)

    except FileNotFoundError as e:
        print(f"\n[Fatal error] No files found for execution: {e}")
        print("Please make sure that all the pre-prepared steps of the project (splitData, run_preprocessing, lightfm_baseline, and embedded_tits) have been completed.")
    except Exception as e:
        print(f"\n[Fatal error] Unexpected error during analysis: {e}")
        raise