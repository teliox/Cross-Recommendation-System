'''
README!

The code combines both book and movie data to learn one integrated recommendation model 
using a library of specialized recommendation systems called LightFM, 
a well-made model, and uses it as a baseline for comparative analysis


Code and Model Usage Example Link
https://github.com/lyst/lightfm
https://www.stepbystepdatascience.com/hybrid-recommender-lightfm-python
'''
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
import pickle
import ast
import re
import os

print("--- Start building LightFM baseline models ---")

# --- 1. Start building LightFM baseline models ---
print("\nStep 1: Loading training Data...")
INPUT_DATA_PATH = './processed_data/splitData/'
try:
    train_movies = pd.read_csv(INPUT_DATA_PATH + 'movielens_train.csv')
    train_books = pd.read_csv(INPUT_DATA_PATH + 'book_train.csv')
    print("-> Finished loading training data.")
except FileNotFoundError as e:
    print(f"[Error] Training data file not found: {e}")
    print("-> Please check if you generated the training/test data by running 'splitData.py ' first.")
    exit()

# --- 2. Data consolidation and feature preparation ---
print("\nStep 2:Data consolidation and feature preparation...")

# Add prefixes to prevent ID conflicts
train_books['item_id_str'] = 'book_' + train_books['book_id'].astype(str)
train_movies['item_id_str'] = 'movie_' + train_movies['movieId'].astype(str)
train_books['user_id_str'] = 'book_' + train_books['user_id'].astype(str)
train_movies['user_id_str'] = 'movie_' + train_movies['userId'].astype(str)

# Interaction data integration
interactions_df = pd.concat([
    train_books[['user_id_str', 'item_id_str', 'rating']],
    train_movies[['user_id_str', 'item_id_str', 'rating']]
]).reset_index(drop=True)

# Consolidate and refine item feature data
item_features_df = pd.concat([
    train_books[['item_id_str', 'genres']],
    train_movies[['item_id_str', 'genres']]
]).drop_duplicates(subset=['item_id_str']).reset_index(drop=True)

# Genre parsing and refining functions (similar to preprocessor.py )
GENRE_MAP = { "children's": "children", 'sci-fi': 'science-fiction' }
def parse_and_clean_genres(genre_str):
    raw_genres = []
    if isinstance(genre_str, str):
        try:
            parsed_obj = ast.literal_eval(genre_str)
            if isinstance(parsed_obj, dict): raw_genres = list(parsed_obj.keys())
            elif isinstance(parsed_obj, list): raw_genres = parsed_obj
        except (ValueError, SyntaxError):
            raw_genres = genre_str.split('|')
    
    cleaned_genres = set()
    for genre in raw_genres:
        split_genres = re.split(r',\s*|,', genre)
        for sub_genre in split_genres:
            clean_genre = sub_genre.lower().strip()
            if clean_genre in GENRE_MAP: clean_genre = GENRE_MAP[clean_genre]
            if clean_genre and clean_genre != '(no genres listed)':
                cleaned_genres.add(clean_genre)
    return list(cleaned_genres)

item_features_df['genre_list'] = item_features_df['genres'].apply(parse_and_clean_genres)
print("-> Data consolidation and feature refinement completed.")


# --- 3. Creating LightFM Dataset Objects ---
print("\nStep 3: Create LightFM Dataset...")
dataset = Dataset()

#mapping user, item ID
dataset.fit(
    users=interactions_df['user_id_str'].unique(),
    items=interactions_df['item_id_str'].unique()
)

# Item Feature (Genre) Mapping
all_genres = set()
for genres in item_features_df['genre_list']:
    all_genres.update(genres)
dataset.fit_partial(item_features=all_genres)
print(f"-> Discovered a total of {len(all_genres)} unique item features (genre).")

# Building an interaction matrix
(interactions, weights) = dataset.build_interactions(
    (row.user_id_str, row.item_id_str, row.rating) for row in interactions_df.itertuples()
)

# Build item feature matrix
item_features = dataset.build_item_features(
    (row.item_id_str, row.genre_list) for row in item_features_df.itertuples()
)
print("-> Complete interaction and feature matrix construction.")

# --- 4. Model Learning ---
print("\nStep 4: Start learning the LightFM model...")
model = LightFM(loss='warp', no_components=50, learning_rate=0.05, random_state=42)
model.fit(interactions, item_features=item_features, epochs=10, num_threads=4, verbose=True)
print("-> Model training completed.")


# --- 5. Save learned models and dataset objects ---
print("\nStep 5: Save learned models and dataset objects...")
output_folder = './processed_data/lightfm/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(output_folder + 'lightfm_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open(output_folder + 'lightfm_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)

print(f"--- LightFM model build and save completed on '{output_folder}' ---")