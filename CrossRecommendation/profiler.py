'''
READ ME!

This code calculates each user's 'preference' as a numeric profile vector 
(User-Genre / User-Decade) based on data preprocessed at preprocessor.py.
'''

import pandas as pd
import numpy as np


'''
It measures which genre is 'rare and special'.
Common genres like drama, romance, and rare genres like noir and war score high


Verbose: 
"Switch that determines whether to output intermediate processes or detailed information 
while the function is running."
'''
def calculate_genre_idf(df, genre_list_col='genre_list', verbose=True):
    if verbose: print(f"\nStart calculating IDF values by genre based on '{genre_list_col}'...")
    
    # --- 1. Calculate Basic Statistics ---
    num_documents = len(df)
    if num_documents == 0: return {}
    
    # --- 2. Calculate Document Frequency for Each Genre ---
    # Flatten the list of genre lists from all items into one large list.
    all_genres_list = [genre for genres in df[genre_list_col].dropna() for genre in genres if genre]
    genre_counts = pd.Series(all_genres_list).value_counts()
       
    # --- 3. Apply the IDF Formula ---
    idf_dict = {genre: np.log(num_documents / (count + 1)) for genre, count in genre_counts.items()}
    if verbose: print("-> IDF calculation completed.")
    return idf_dict

'''
Create User-genre Profile
The genre information of all items viewed by the user is collected, 
and the rating left by the user for each genre is weighted to average it.
'''
def create_user_profile_matrix(df, id_col, rating_col, genre_df, verbose=True):
    if verbose: print(f"\nStart generating baseline profile matrix '{id_col}'...")
    
    # --- Define the Weighted Average Calculation Function ---
    def weighted_avg(g):
        ratings = g[rating_col]
        vectors = genre_df.loc[g.index]
        
        # Calculate only if the sum of ratings is not zero (to prevent division by zero).
        return np.average(vectors, axis=0, weights=ratings) if ratings.sum() != 0 else np.zeros(vectors.shape[1])

    # df.groupby(id_col) is like looping through each user.
    # .apply(weighted_avg) executes the weighted_avg function for each user group.
    user_profile = df.groupby(id_col).apply(weighted_avg)
    user_profile_df = pd.DataFrame(user_profile.tolist(), columns=genre_df.columns, index=user_profile.index)
    
    if verbose: print("-> Complete creating baseline profile matrix.")
    return user_profile_df

'''
The IDF value calculated in the User-Genre profile is once again weighted to reflect the 'importance of genre' and ensemble the two weights
'''
def create_user_profile_matrix_with_idf(df, id_col, rating_col, genre_df, genre_idf_map, verbose=True):
    if verbose: print(f"\nStart generating the base IDF weighted profile matrix '{id_col}'...")
    
    # Create a vector of IDF values ordered identically to the columns of the genre DataFrame.
    idf_vector = np.array([genre_idf_map.get(genre.strip(), 0) for genre in genre_df.columns])
    
    def weighted_avg_with_idf(g):
        ratings = g[rating_col]
        
        # Multiply the genre one-hot vectors of each item by the IDF vector.
        # This increases the influence of rarer genres.
        weighted_genre_vectors = genre_df.loc[g.index] * idf_vector
        return np.average(weighted_genre_vectors, axis=0, weights=ratings) if ratings.sum() != 0 else np.zeros(weighted_genre_vectors.shape[1])

    # --- Group by User and Convert to DataFrame ---
    user_profile = df.groupby(id_col).apply(weighted_avg_with_idf)
    user_profile_df = pd.DataFrame(user_profile.tolist(), columns=genre_df.columns, index=user_profile.index)
    
    if verbose: print("-> IDF weighted profile matrix generation completed.")
    return user_profile_df

'''
Use the previously created decade feature to create a period preference profile (User-Decade)

For example, a person who likes Hong Kong romance movies from the 70s is more likely to be recommended romance movies that fit the era
'''
def create_user_decade_profile_matrix(df, id_col, rating_col, verbose=True):
    if verbose: print(f"\nStart generating a decade profile matrix based on '{id_col}'...")

    # Converts the 'decade' column into columns like '1980s', '1990s', etc.
    decade_onehot_df = pd.get_dummies(df['decade'])
    
    def weighted_avg_decade(g):
        ratings = g[rating_col]
        vectors = decade_onehot_df.loc[g.index] # The decade one-hot vectors for the items this user rated.
        return np.average(vectors, axis=0, weights=ratings) if ratings.sum() != 0 else np.zeros(vectors.shape[1])
    
    
    # --- Group by User and Convert to DataFrame ---
    user_profile = df.groupby(id_col).apply(weighted_avg_decade)
    user_profile_df = pd.DataFrame(user_profile.tolist(), columns=decade_onehot_df.columns, index=user_profile.index)
    
    if verbose: print("->Decade Profile Matrix Generation Completed.")
    return user_profile_df