'''
README!

Hyperparameter Optimization (Genre weight(Genre profile with idf), Semantic weight(S-BERT))
with Grid Search

result : 0.5 / 0.5 is best performance
'''

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# --- 0. Loading Modules and Models ---
from recommender import CrossDomainRecommender
from lightfm import LightFM

# --- 1. Path and hyperparameter candidate settings ---
# (Set the same path as main.py)
SPLIT_DATA_PATH = './processed_data/splitData/'
PROFILE_PATH = './processed_data/profile/'
LIGHTFM_FOLDER = './processed_data/lightfm/'
BOOK_TRAIN_PATH = SPLIT_DATA_PATH + 'book_train.csv'
MOVIE_TRAIN_PATH = SPLIT_DATA_PATH + 'movielens_train.csv'
BOOK_TEST_PATH = SPLIT_DATA_PATH + 'book_test.csv'
MOVIE_DATA_PATH = 'movielens_preprocessed_final.csv'
BOOK_GENRE_PROFILE_PATH = PROFILE_PATH + 'book_user_genre_profile_ensemble.csv'
MOVIE_GENRE_PROFILE_PATH = PROFILE_PATH + 'movie_user_genre_profile_ensemble.csv'
BOOK_DECADE_PROFILE_PATH = PROFILE_PATH + 'book_user_decade_profile.csv'
MOVIE_DECADE_PROFILE_PATH = PROFILE_PATH + 'movie_user_decade_profile.csv'
LIGHTFM_MODEL_PATH = LIGHTFM_FOLDER + 'lightfm_model.pkl'
LIGHTFM_DATASET_PATH = LIGHTFM_FOLDER + 'lightfm_dataset.pkl'

# Hyperparameter candidates for Grid Search
GENRE_WEIGHT_CANDIDATES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
SEMANTIC_WEIGHT_CANDIDATES = [0.1, 0.3, 0.5, 0.6, 0.8]
NUM_TEST_USERS = 100 # Number of sample users to control test time

# --- 2. LightFM helper function ---
def get_lightfm_vectors(model, dataset_mapping, ids, id_prefix):
    """Extract the latent vectors of specific IDs from the LightFM model."""
    _, item_embeddings = model.get_item_representations(features=None)
    indices = [dataset_mapping.get(id_prefix + str(i)) for i in ids if (id_prefix + str(i)) in dataset_mapping]
    if not indices: return None
    return item_embeddings[indices]

# --- 3. Main Grid Search Function ---
def run_grid_search():
    """It traverses hyperparameter combinations and performs Grid Search to find the optimal value."""
    print("======= Start Hyperparameter Grid Search =======")
    
    # --- Model and data loading (once only) ---
    try:
        with open(LIGHTFM_MODEL_PATH, 'rb') as f: lightfm_model = pickle.load(f)
        with open(LIGHTFM_DATASET_PATH, 'rb') as f: lightfm_dataset = pickle.load(f)
        _, _, item_mapping, _ = lightfm_dataset.mapping()
        book_test_df = pd.read_csv(BOOK_TEST_PATH)
        print("-> Default data loading completed for Grid Search.")
    except Exception as e:
        print(f"[Fatal error] Failed to load files while preparing Grid Search: {e}")
        return

    all_results = []
    
    # --- Grid Search Loop ---
    for gw in GENRE_WEIGHT_CANDIDATES:
        for sw in SEMANTIC_WEIGHT_CANDIDATES:
            print(f"\n\n--- Start test: GENRE_WEIGHT={gw}, SEMANTIC_WEIGHT={sw} ---")
            
            # 1. Initialize the recommended engine with the current parameter
            recommender_engine = CrossDomainRecommender(
                book_genre_profile_path=BOOK_GENRE_PROFILE_PATH,
                movie_genre_profile_path=MOVIE_GENRE_PROFILE_PATH,
                book_decade_profile_path=BOOK_DECADE_PROFILE_PATH,
                movie_decade_profile_path=MOVIE_DECADE_PROFILE_PATH,
                book_data_path=BOOK_TRAIN_PATH,
                movie_data_path=MOVIE_TRAIN_PATH,
                genre_weight=gw,
                semantic_weight=sw
            )
            
            # 2. Perform evaluation
            test_user_ids = book_test_df['user_id'].unique()
            average_similarities = []
            
            for user_id in tqdm(test_user_ids[:NUM_TEST_USERS], desc=f"Validating (GW={gw}, SW={sw})"):
                recommendations_df, _ = recommender_engine.recommend_movies(user_id, top_n=10)
                if recommendations_df is None or recommendations_df.empty: continue
                
                user_liked_books = book_test_df[(book_test_df['user_id'] == user_id) & (book_test_df['rating'] >= 4)]['book_id'].tolist()
                if not user_liked_books: continue
                
                liked_books_vectors = get_lightfm_vectors(lightfm_model, item_mapping, user_liked_books, 'book_')
                if liked_books_vectors is None: continue
                actual_taste_vector = liked_books_vectors.mean(axis=0).reshape(1, -1)
                
                recommended_movie_ids = recommendations_df.index.tolist()
                recommended_movies_vectors = get_lightfm_vectors(lightfm_model, item_mapping, recommended_movie_ids, 'movie_')
                if recommended_movies_vectors is None: continue
                
                sims = cosine_similarity(actual_taste_vector, recommended_movies_vectors)[0]
                average_similarities.append(np.mean(sims))
            
            final_avg_similarity = np.mean(average_similarities) if average_similarities else 0
            
            # 3. Save the result
            all_results.append({
                'genre_weight': gw,
                'semantic_weight': sw,
                'avg_similarity_score': final_avg_similarity
            })

    # --- Final Results Output ---
    print("\n\n======= Grid Search Final Results =======")
    results_df = pd.DataFrame(all_results)
    print(results_df.sort_values('avg_similarity_score', ascending=False))
    
    if not results_df.empty:
        best_params = results_df.loc[results_df['avg_similarity_score'].idxmax()]
        print("\n--- Best hyperparameter combination ---")
        print(best_params)

if __name__ == '__main__':
    run_grid_search()