import pandas as pd
import time
import re

# --- 0. Settings ---
# Please specify the folder path where the .dat files are located.
DATA_FOLDER = '/Users/kimtaeryang/Desktop/VSC/ML/MachineLearning/TermProject/dataset/ml-1m'
# The file name to save the final result
FINAL_OUTPUT_FILE = 'movielens_preprocessed_final.csv' 

print("======= Starting Full Preprocessing for MovieLens Data =======")

# --- 1. Data Loading ---
print("\nStep 1: Starting to load original .dat files...")
start_time = time.time()

# Loading ratings.dat (only userId, movieId, rating are needed)
r_cols = ['userId', 'movieId', 'rating', 'timestamp']
df_ratings = pd.read_csv(
    DATA_FOLDER + 'ratings.dat', 
    sep='::', 
    header=None,
    names=r_cols, 
    engine='python',
    encoding='latin-1'
)
df_ratings = df_ratings.drop('timestamp', axis=1)

# Loading movies.dat (only movieId, title, genres are needed)
m_cols = ['movieId', 'title', 'genres']
df_movies = pd.read_csv(
    DATA_FOLDER + 'movies.dat', 
    sep='::', 
    header=None,
    names=m_cols, 
    engine='python',
    encoding='latin-1'
)
print(f"-> Data loading complete. Time elapsed: {time.time() - start_time:.2f} seconds")

# --- 2. Merging DataFrames ---
print("\nStep 2: Starting to merge ratings data with movie metadata...")
start_time = time.time()
df_merged = pd.merge(df_ratings, df_movies, on='movieId', how='inner')
print(f"-> Merge complete. (Total ratings: {len(df_merged)}) - Time elapsed: {time.time() - start_time:.2f} seconds")

# --- 3. Transforming 'genres' Column to Dictionary Format ---
print("\nStep 3: Starting 'genres' column format conversion...")
start_time = time.time()

def genres_to_dict(genre_str):
    if pd.isna(genre_str) or genre_str == '' or genre_str == '(no genres listed)':
        return {}
    genres_list = genre_str.split('|')
    return {genre: 1 for genre in genres_list}

df_merged['genres'] = df_merged['genres'].apply(genres_to_dict)
print(f"-> Genre conversion complete. Time elapsed: {time.time() - start_time:.2f} seconds")

# --- 4. Extracting 'year' and 'cleaned_title' from 'title' ---
print("\nStep 4: Starting extraction of year and clean title from 'title'...")
start_time = time.time()

def extract_year_from_title(title):
    if not isinstance(title, str): return None
    match = re.search(r'\((\d{4})\)', title)
    if match: return int(match.group(1))
    return None

def clean_title(title):
    if not isinstance(title, str): return title
    return re.sub(r'\s*\(\d{4})\s*$', '', title).strip()

df_merged['year'] = df_merged['title'].apply(extract_year_from_title)
df_merged['title_cleaned'] = df_merged['title'].apply(clean_title)
print(f"-> Year and clean title extraction complete. Time elapsed: {time.time() - start_time:.2f} seconds")


# --- 5. Final Result Check and Save ---
print("\n--- Final Preprocessing Result ---")

# Reorder columns for better readability
final_cols = [
    'userId', 'movieId', 'rating', 
    'title_cleaned', 'year', 'genres', 
    'title' # Keep the original title for reference at the end
]
df_final = df_merged[final_cols]

print("Information of the final generated DataFrame:")
print(df_final.info())
print("\nTop 5 data samples:")
print(df_final.head())

print(f"\nSaving the integrated DataFrame to '{FINAL_OUTPUT_FILE}'...")
start_time = time.time()
df_final.to_csv(FINAL_OUTPUT_FILE, index=False)
print(f"Save complete! - Time elapsed: {time.time() - start_time:.2f} seconds")

print("\n======= Full Preprocessing for MovieLens Data Complete! =======")