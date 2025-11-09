'''
README!

Split entire dataset into training and test set
'''
import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("--- Start separating training/test data ---")

# --- 0. Setting the Path ---
INPUT_MOVIE_PATH = './processed_data/finalDataset/movielens_preprocessed_final.csv'
INPUT_BOOK_PATH = './processed_data/finalDataset/final_book_data.csv'

OUTPUT_FOLDER = './processed_data/splitData/'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# --- 1. Separate MovieLens data ---
print("\n[MovieLens Data Processing]")
try:
    df_movies = pd.read_csv(INPUT_MOVIE_PATH)
    
    # Calculate how many times each userId appears
    user_counts_movies = df_movies['userId'].value_counts()
    
    # Filter only userId that appears more than once
    active_movie_users = user_counts_movies[user_counts_movies >= 2].index
    
    # Generating filtered data
    df_movies_filtered = df_movies[df_movies['userId'].isin(active_movie_users)]
    
    removed_users_count = len(user_counts_movies) - len(active_movie_users)
    if removed_users_count > 0:
        print(f"->We excluded {removed_users_count} movie users with one review from the analysis.")
    
    # Run train_test_split with filtered data
    train_movies, test_movies = train_test_split(
        df_movies_filtered,
        test_size=0.2,
        random_state=42,
        stratify=df_movies_filtered['userId'] # Use the stratify option
    )
    
    # Store separated data
    train_movies.to_csv(OUTPUT_FOLDER + 'movielens_train.csv', index=False)
    test_movies.to_csv(OUTPUT_FOLDER + 'movielens_test.csv', index=False)
    print(f"-> MovieLens disconnected: {len(train_movies)} for learning, {len(test_movies)} for testing")

except FileNotFoundError:
    print(f"[Error] file '{INPUT_MOVIE_PATH}' not found, please run the previous step first.")


# --- 2. Separate Goodreads data ---
print("\n[Goodreads Data Processing]")
try:
    df_books = pd.read_csv(INPUT_BOOK_PATH)
    
    # Calculate how many times each user_id appears
    user_counts_books = df_books['user_id'].value_counts()
    
    # Filter only user_id that appears more than once
    active_book_users = user_counts_books[user_counts_books >= 2].index
    
    # Generating filtered data
    df_books_filtered = df_books[df_books['user_id'].isin(active_book_users)]
    
    removed_users_count = len(user_counts_books) - len(active_book_users)
    if removed_users_count > 0:
        print(f"-> We excluded {removed_users_count} book users with one review from the analysis.")
    
    # Run train_test_split with filtered data
    train_books, test_books = train_test_split(
        df_books_filtered,
        test_size=0.2,
        random_state=42,
        stratify=df_books_filtered['user_id'] # Use the stratify option
    )
    
    # Store separated data
    train_books.to_csv(OUTPUT_FOLDER + 'book_train.csv', index=False)
    test_books.to_csv(OUTPUT_FOLDER + 'book_test.csv', index=False)
    print(f"-> Goodreads Isolated: {len(train_books)} for learning, {len(test_books)} for testing")

except FileNotFoundError:
    print(f"[Error] file '{INPUT_BOOK_PATH}' not found, please run the previous step first.")


print("\n--- Complete all data separation ---")