import pandas as pd
import time

def main():
    print("======= Starting Final Book Dataset Creation for Modeling =======")
    start_time_total = time.time()
    
    # --- 1. File Path Definitions ---
    reviews_original_path = '/Users/kimtaeryang/Desktop/VSC/ML/MachineLearning/TermProject/dataset/goodreads/goodreads_reviews_dedup.json.gz'
    meta_full_path = 'book_meta_full_final.csv'
    output_path = 'final_book_data.csv'
    
    # --- 2. Review Data Sampling ---
    print("\nStep 1: Starting review data sampling...")
    start_time = time.time()
    
    # Read the first chunk of 1,000,000 lines from the file
    chunk_iterator = pd.read_json(reviews_original_path, lines=True, compression='gzip', chunksize=1000000)
    df_reviews_sample = next(chunk_iterator)
    df_reviews_sample = df_reviews_sample[['user_id', 'book_id', 'rating']]
    
    print(f"-> {len(df_reviews_sample)} reviews sampled successfully. Time elapsed: {time.time() - start_time:.2f} seconds")

    # --- 3. Loading Integrated Metadata ---
    print("\nStep 2: Loading integrated metadata...")
    start_time = time.time()
    try:
        df_meta_full = pd.read_csv(meta_full_path)
        print(f"-> Metadata loading complete. Time elapsed: {time.time() - start_time:.2f} seconds")
    except FileNotFoundError:
        print(f"[ERROR] '{meta_full_path}' not found. Please run 'create_book_metadata.py' first.")
        return

    # --- 4. Merging Data ---
    print("\nStep 3: Merging review samples with metadata...")
    start_time = time.time()
    
    # Unify book_id types to string for a safe merge
    df_reviews_sample['book_id'] = df_reviews_sample['book_id'].astype(str)
    df_meta_full['book_id'] = df_meta_full['book_id'].astype(str)
    
    df_final = pd.merge(df_reviews_sample, df_meta_full, on='book_id', how='left')
    print(f"-> Merge complete. (Final rows: {len(df_final)}) - Time elapsed: {time.time() - start_time:.2f} seconds")
    
    # --- 5. Handling Missing Values and Saving ---
    missing_before = df_final['title'].isnull().sum()
    print(f"\nNumber of reviews with missing title/genre info after merge: {missing_before}")
    
    # Drop rows with no title or genres, as they are not useful for profile creation
    df_final.dropna(subset=['title', 'genres'], inplace=True)
    print(f"Final number of rows after dropping missing values: {len(df_final)}")

    df_final.to_csv(output_path, index=False)
    print(f"\nThe final dataset has been saved to '{output_path}'.")
    print(f"Total time elapsed: {time.time() - start_time_total:.2f} seconds")

if __name__ == '__main__':
    main()