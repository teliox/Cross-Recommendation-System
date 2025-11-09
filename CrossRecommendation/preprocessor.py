'''
README!

From different datasets of books and movies, genre columns are extracted to make them clean and unified.
The year information is extracted and processed into a chronological feature.

Method:
From different datasets of books and movies, genre columns are extracted to make them clean and unified.
Year information is extracted and processed into a decade feature.

Method:
It's a genre that's in the dictionary form
Using 'ast.literal_val(x).keys()'
Return to the list.

Spelling is similar, but the same genre types are the same for the book and genre
It is processed according to the predefined mapping rule

The year is chunked every 10 years to create a decade feature

Hand over the results to profiler.py to create a user-genre/user-decade profile.

MultiLabelBinarizer() :
Unlike traditional one-hot encoding methods,
One book or movie can have multiple genres, so I used MultiLabelBinarizer()!
'''
import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
import re


# rules for unifying different or similar genres of spelling
GENRE_MAP = {
    # the uniformity of spelling/expression
    "children's": "children",
    "childrens": "children", 
    "sci-fi": "science-fiction",
    "film-noir": "noir",
    
    # semantic integration (optional, not currently used)
    # 'historical fiction': 'history',
    # 'biography': 'history',
}

def clean_and_split_genres(genre_list):
    """
    It receives a list of unrefined genres, separates the tied genres, and unifies the spelling.
    Ex: ['fantasy, paranormal', "Children's"] -> ['fantasy', 'paranormal', 'children']
    """
    cleaned_genres = set()
    for genre in genre_list:
        # Separate genres grouped by comma(,) or comma+blank(,)
        split_genres = re.split(r',\s*|,', genre)
        
        for sub_genre in split_genres:
            clean_genre = sub_genre.lower().strip()
            
            # Use GENRE_MAP to unify spelling
            if clean_genre in GENRE_MAP:
                clean_genre = GENRE_MAP[clean_genre]
            
            if clean_genre and clean_genre != '(no genres listed)':
                cleaned_genres.add(clean_genre)
                
    return sorted(list(cleaned_genres))

def parse_genres_from_string(df, genre_col='genres', verbose=True):
    """
    Pars and refine various types of genre strings to create a 'genre_list' column.
    """
    raw_genre_list_col = 'raw_genre_list' # Temporary column
    parsed = False

    # Verify that the data has a valid genre string
    valid_genres = df[genre_col].dropna()
    if valid_genres.empty:
        df['genre_list'] = [[] for _ in range(len(df))]
        return df

    sample_genre = valid_genres.iloc[0]
    
    # Try ast.literal_val first (dictionary or list string)
    try:
        parsed_sample = ast.literal_eval(sample_genre)
        if isinstance(parsed_sample, dict):
            if verbose: print(f"-> Column '{genre_col}': detect dictionary string type...")
            def to_genre_keys(x):
                if pd.isna(x): return []
                try: return list(ast.literal_eval(x).keys())
                except: return []
            df[raw_genre_list_col] = df[genre_col].apply(to_genre_keys)
            parsed = True
    except (ValueError, SyntaxError):
        pass # Moving on to pipe treatment

    # If not parsed, treat with pipe (|) separator
    if not parsed:
        if verbose: print(f"->Column '{genre_col}': pipe(|) delimited string type detected...")
        df[raw_genre_list_col] = df[genre_col].str.split('|')
        df[raw_genre_list_col] = df[raw_genre_list_col].fillna('').apply(lambda x: x if isinstance(x, list) else [])

    if verbose: print("-> Performing genre separation and spelling unification tasks...")
    df['genre_list'] = df[raw_genre_list_col].apply(clean_and_split_genres)
    
    if raw_genre_list_col in df.columns:
        df = df.drop(raw_genre_list_col, axis=1)
    
    return df

def create_decade_feature(df, verbose=True):
    """Create a 'decade' column based on the 'year' column."""
    if verbose: print("-> Creating 'decade' features...")
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    def year_to_decade(y):
        if pd.isna(y): return 'Unknown'
        return f"{int(y // 10) * 10}s"
    df['decade'] = df['year'].apply(year_to_decade)
    return df

def preprocess_data(df, is_movie=True, verbose=True):
    """
    Performs all preprocessing operations on a single data frame, 
    and returns processed df and genre one-hot encoding df.
    """
    if verbose: print(f"\n--- {'Movie' if is_movie else 'Book'} Data preprocessing starts ---")
    
    df_processed = df.copy()
    
    # 1. Genre parsing and refining
    df_processed = parse_genres_from_string(df_processed, genre_col='genres', verbose=verbose)
    
    # 2. Create a Decade Feature
    if 'year' in df_processed.columns:
        df_processed = create_decade_feature(df_processed, verbose=verbose)
    
    # 3. Genre One-Hot Encoding
    mlb = MultiLabelBinarizer()
    # Prepare if genre_list is empty
    valid_genre_lists = df_processed['genre_list'][df_processed['genre_list'].apply(lambda x: len(x) > 0)]
    if valid_genre_lists.empty:
        # If all books/movies have no genre
        genre_df = pd.DataFrame(index=df_processed.index)
    else:
        genre_onehot = mlb.fit_transform(df_processed['genre_list'])
        genre_df = pd.DataFrame(genre_onehot, columns=mlb.classes_, index=df_processed.index)
    
    if verbose: print(f"-> Preprocessed. Discovered {len(genre_df.columns)} unique genres.")
    return df_processed, genre_df

def align_genre_space(movie_genre_df, book_genre_df, verbose=True):
    """Unify the columns (genre space) of the two genre data frames."""
    if verbose: print("\n--- Genre space unification begins ---")
    
    total_genres = sorted(set(movie_genre_df.columns).union(set(book_genre_df.columns)))
    if verbose: print(f"-> Discovered a total of {len(total_genres)} integrated genres.")
    
    movie_total_genre_df = movie_genre_df.reindex(columns=total_genres, fill_value=0)
    book_total_genre_df = book_genre_df.reindex(columns=total_genres, fill_value=0)
    
    if verbose: print("-> Genre space unification completed.")

    return movie_total_genre_df, book_total_genre_df, total_genres
