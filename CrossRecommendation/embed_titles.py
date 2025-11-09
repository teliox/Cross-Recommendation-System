'''
README!

The code uses SBERT to embed and store the titles of each book and movie, 
allowing you to measure semantic similarity.
Later, it re-ranks the recommended list with similarity values
'''

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import time

# --- setting ---
#########################################################################################
PROCESSED_DATA = './processed_data/'
BOOK_DATA_PATH = PROCESSED_DATA + 'splitData/book_train.csv'
MOVIE_DATA_PATH = PROCESSED_DATA + 'splitData/movielens_train.csv'

# S-BERT Model Selection (Light and Performance Model)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Output file path
PROCESSED_DATA_PATH = './processed_data/sbert/'
BOOK_EMBEDDINGS_PATH = PROCESSED_DATA_PATH + '/book_title_embeddings.npy'
BOOK_ID_MAP_PATH = PROCESSED_DATA_PATH + '/book_id_map.csv'
MOVIE_EMBEDDINGS_PATH = PROCESSED_DATA_PATH + '/movie_title_embeddings.npy'
MOVIE_ID_MAP_PATH = PROCESSED_DATA_PATH + '/movie_id_map.csv'

#########################################################################################

def main():
    print(f"======= Start creating title embeddings (Model): {MODEL_NAME}) =======")
    
    # 1. Loading S-BERT Model
    print("Loading S-BERT model...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 2. Data loading and unique title extraction
    df_books = pd.read_csv(BOOK_DATA_PATH)[['book_id', 'title']].drop_duplicates().reset_index(drop=True)
    df_movies = pd.read_csv(MOVIE_DATA_PATH)[['movieId', 'title_cleaned']].drop_duplicates().reset_index(drop=True)
    
    # 3. Book title embedding
    print(f"\nStart embeddings {len(df_books)}book titles...")
    start_time = time.time()
    book_embeddings = model.encode(df_books['title'].tolist(), show_progress_bar=True)
    print(f"-> Book embedding completed. Time required: {time.time() - start_time:.2f}seconds")
    
    # 4. Movie title embedding
    print(f"\nStart embeddings {len(df_movies)}movie titles...")
    start_time = time.time()
    movie_embeddings = model.encode(df_movies['title_cleaned'].tolist(), show_progress_bar=True)
    print(f"-> Movie Embedding completed. Time Required: {time.time() - start_time:.2f}seconds")
    
    # 5. Save embedding results and ID mapping
    np.save(BOOK_EMBEDDINGS_PATH, book_embeddings)
    np.save(MOVIE_EMBEDDINGS_PATH, movie_embeddings)
    
    # Stores a file that maps the index of the embedding array to the real ID.
    df_books[['book_id']].to_csv(BOOK_ID_MAP_PATH, index=True) # index is the order of the array
    df_movies[['movieId']].to_csv(MOVIE_ID_MAP_PATH, index=True)

    print(f"\nAll embeddings and ID maps were saved in folder '{PROCESSED_DATA_PATH}'.")

if __name__ == '__main__':
    main()