import pandas as pd
import os
import time

# --- Module Import ---
from preprocessor import preprocess_data, align_genre_space
from profiler import (
    calculate_genre_idf, 
    create_user_profile_matrix, 
    create_user_profile_matrix_with_idf,
    create_user_decade_profile_matrix
)

# --- Setting ---
# [Correction] The input data uses only the training data in the 'splitData' folder.
INPUT_DATA_PATH = './processed_data/splitData/'
BOOK_DATA_PATH = INPUT_DATA_PATH + 'book_train.csv'
MOVIE_DATA_PATH = INPUT_DATA_PATH + 'movielens_train.csv'

# Output Profile Data Path
PROFILE_DATA_PATH = './processed_data/profile/'

# Define the entire path of the final profile file
BOOK_GENRE_PROFILE_PATH = PROFILE_DATA_PATH + 'book_user_genre_profile_ensemble.csv'
MOVIE_GENRE_PROFILE_PATH = PROFILE_DATA_PATH + 'movie_user_genre_profile_ensemble.csv'
BOOK_DECADE_PROFILE_PATH = PROFILE_DATA_PATH + 'book_user_decade_profile.csv'
MOVIE_DECADE_PROFILE_PATH = PROFILE_DATA_PATH + 'movie_user_decade_profile.csv'

# Model hyperparameter
ENSEMBLE_ALPHA = 0.7 

def main():
    start_time_total = time.time()
    print("======= Start full data preprocessing and profile creation pipeline =======")

    # --- 1. Data Loading ---
    print("\n--- 1. Loading Training Data ---")
    try:
        df_books_raw = pd.read_csv(BOOK_DATA_PATH)
        df_movies_raw = pd.read_csv(MOVIE_DATA_PATH)
        print("-> Complete loading of book/movie training data.")
    except FileNotFoundError as e:
        print(f"[Fatal error] Input data file not found: {e}")
        print("-> Please check if you generated the training/test data by running 'splitData.py ' first.")
        return # Ending the program

    # --- 2. Pre-processing (call preprocessor.py) ---
    # The preprocess_data function internally handles all genre parsing, solidarity feature generation, and so on.
    df_books, book_genre_df = preprocess_data(df_books_raw, is_movie=False)
    df_movies, movie_genre_df = preprocess_data(df_movies_raw, is_movie=True)
    
    # --- 3. Create Profile (call profiler.py) ---
    print("\n--- 3. Create a user profile matrix ---")
    
    # 3-1. Genre space unification
    movie_total_genre_df, book_total_genre_df, total_genres = align_genre_space(
        movie_genre_df, book_genre_df
    )
    
    # 3-2. Create IDF Map
    temp_combined_df = pd.concat([df_books[['genre_list']], df_movies[['genre_list']]], ignore_index=True)
    total_idf_map = calculate_genre_idf(temp_combined_df, genre_list_col='genre_list')

    # 3-3. Create a genre ensemble profile
    book_profile_A = create_user_profile_matrix(df_books, 'user_id', 'rating', book_total_genre_df)
    movie_profile_A = create_user_profile_matrix(df_movies, 'userId', 'rating', movie_total_genre_df)
    
    book_profile_B = create_user_profile_matrix_with_idf(df_books, 'user_id', 'rating', book_total_genre_df, total_idf_map)
    movie_profile_B = create_user_profile_matrix_with_idf(df_movies, 'userId', 'rating', movie_total_genre_df, total_idf_map)
    
    book_user_genre_profile = (ENSEMBLE_ALPHA * book_profile_A) + ((1 - ENSEMBLE_ALPHA) * book_profile_B)
    movie_user_genre_profile = (ENSEMBLE_ALPHA * movie_profile_A) + ((1 - ENSEMBLE_ALPHA) * movie_profile_B)
    print("-> Genre Ensemble Profile Created.")
    
    # 3-4. Create a Decade Profile
    # 'decade' column has already been created in the preprocessor, so it is available immediately
    book_user_decade_profile = create_user_decade_profile_matrix(df_books, 'user_id', 'rating')
    movie_user_decade_profile = create_user_decade_profile_matrix(df_movies, 'userId', 'rating')
    
    # --- 4. Save Profile ---
    print("\n--- 4. Save Profile Matrix ---")
    if not os.path.exists(PROFILE_DATA_PATH):
        os.makedirs(PROFILE_DATA_PATH)
        print(f"-> Directory '{PROFILE_DATA_PATH}' created.")

    book_user_genre_profile.to_csv(BOOK_GENRE_PROFILE_PATH)
    movie_user_genre_profile.to_csv(MOVIE_GENRE_PROFILE_PATH)
    book_user_decade_profile.to_csv(BOOK_DECADE_PROFILE_PATH)
    movie_user_decade_profile.to_csv(MOVIE_DECADE_PROFILE_PATH)
    
    print(f"-> All profile files have been successfully saved to '{PROFILE_DATA_PATH}'.")
    
    print("\n======= Pre-processing and profile creation complete! =======")
    print(f"Total time taken: {time.time() - start_time_total:.2f} seconds")

if __name__ == '__main__':
    main()