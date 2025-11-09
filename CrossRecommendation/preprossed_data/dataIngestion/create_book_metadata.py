import pandas as pd
import gc
import time

def process_large_json_in_chunks(filepath, columns_to_keep, chunksize=500000):
    """
    Reads a large JSON.gz file in chunks, extracts necessary columns, and returns a single DataFrame.
    """
    print(f"Processing file '{filepath}'...")
    chunk_iterator = pd.read_json(filepath, lines=True, compression='gzip', chunksize=chunksize)
    processed_chunks = []
    
    for i, chunk in enumerate(chunk_iterator):
        chunk_subset = chunk[columns_to_keep]
        processed_chunks.append(chunk_subset)
        print(f"  - Chunk {i+1} processed.")
        del chunk, chunk_subset
        gc.collect()
        
    final_df = pd.concat(processed_chunks, ignore_index=True)
    print(f"Finished processing file '{filepath}'.")
    return final_df

def main():
    print("======= Starting Integrated Book Metadata Creation =======")
    start_time_total = time.time()

    # --- 1. File Paths and Column Definitions ---
    books_meta_path = '/Users/kimtaeryang/Desktop/VSC/ML/MachineLearning/TermProject/dataset/goodreads/goodreads_books.json.gz'
    books_meta_cols = ['book_id', 'title', 'average_rating', 'ratings_count', 'publication_year']

    genres_meta_path = '/Users/kimtaeryang/Desktop/VSC/ML/MachineLearning/TermProject/dataset/goodreads/goodreads_book_genres_initial.json.gz'
    genres_meta_cols = ['book_id', 'genres']
    
    output_path = 'book_meta_full_final.csv'

    # --- 2. Process Each Metadata File ---
    df_books_meta = process_large_json_in_chunks(books_meta_path, books_meta_cols)
    df_genres_meta = process_large_json_in_chunks(genres_meta_path, genres_meta_cols)

    # --- 3. Unify Data Types and Merge ---
    print("\nStarting to merge two metadata DataFrames...")
    df_books_meta['book_id'] = df_books_meta['book_id'].astype(str)
    df_genres_meta['book_id'] = df_genres_meta['book_id'].astype(str)
    
    df_meta_full = pd.merge(df_books_meta, df_genres_meta, on='book_id', how='left')
    print("Merge complete.")

    # --- 4. Clean 'year' Column ---
    print("\nCleaning 'year' column...")
    df_meta_full['year'] = pd.to_numeric(df_meta_full['publication_year'], errors='coerce')
    df_meta_full['year'] = df_meta_full['year'].astype('Int64')
    df_meta_full = df_meta_full.drop('publication_year', axis=1)
    print("Cleaning complete.")

    # --- 5. Final Result Check and Save ---
    print("\nFinal integrated metadata information:")
    print(df_meta_full.info())
    print("\nTop 5 data samples:")
    print(df_meta_full.head())

    df_meta_full.to_csv(output_path, index=False)
    print(f"\nAll tasks complete! The final result has been saved to '{output_path}'.")
    print(f"Total time elapsed: {time.time() - start_time_total:.2f} seconds")

if __name__ == '__main__':
    main()