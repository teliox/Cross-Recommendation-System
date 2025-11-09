import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CrossDomainRecommender:
    """
    Final recommendation engine with added stability and result reinforcement utilizing genre, solidarity, and S-BERT.
    """
    def __init__(self, book_genre_profile_path, movie_genre_profile_path,
                 book_decade_profile_path, movie_decade_profile_path,
                 book_data_path, movie_data_path,
                 genre_weight=0.8, semantic_weight=0.15):
        """
        Initialize the recommender.
        """
        print("--- Start Recommendation Engine Initialization ---")
        self.genre_weight = genre_weight
        self.decade_weight = 1 - genre_weight
        self.semantic_weight = semantic_weight

        try:
            # Loading Profiles
            # These profiles are pre-calculated based on user preferences for genres and decades.
            # Filtering Technique: These profiles are the output of a Content-based approach.
            self.book_genre_profile_df = pd.read_csv(book_genre_profile_path, index_col=0)
            self.movie_genre_profile_df = pd.read_csv(movie_genre_profile_path, index_col=0)
            self.book_decade_profile_df = pd.read_csv(book_decade_profile_path, index_col=0)
            self.movie_decade_profile_df = pd.read_csv(movie_decade_profile_path, index_col=0)

            # The original training data is needed for various calculations (e.g., neighbor ratings).
            # Save these data frames to self for evaluator to use
            self.book_train_df = pd.read_csv(book_data_path)
            self.movie_train_df = pd.read_csv(movie_data_path)
            # Maintain existing variable names (because they are used by internal logic)
            self.book_data_df = self.book_train_df
            self.movie_data_df = self.movie_train_df
            
            # --- Finalize Initialization --
            # Rest of the initialization process
            self._align_profiles()
            self._load_embeddings()
            self._prepare_popularity_fallback()
            print("-> Complete all initializations.")

        except FileNotFoundError as e:
            print(f"[Fatal error] Required file loading failed: {e}")
            raise
            
    '''
    Since book profiles and movie profiles may have different 'genre' or 'decade' lists, 
    it forces the two profiles to have the same column.
    !!!Very important!!!
    '''
    def _align_profiles(self):
        # Align genre profile columns
        common_genres = self.book_genre_profile_df.columns.intersection(self.movie_genre_profile_df.columns)
        self.book_genre_profile_df = self.book_genre_profile_df[common_genres]
        self.movie_genre_profile_df = self.movie_genre_profile_df[common_genres]
        
        # Align decade profile columns
        common_decades = self.book_decade_profile_df.columns.intersection(self.movie_decade_profile_df.columns)
        self.book_decade_profile_df = self.book_decade_profile_df[common_decades]
        self.movie_decade_profile_df = self.movie_decade_profile_df[common_decades]

    '''
    Bring up the pre-calculated S-BERT title vectors and prepare the recommendation system 
    to understand the 'semantic of the title'
    '''
    def _load_embeddings(self):
        try:
            self.book_embeddings = np.load('./processed_data/sbert/book_title_embeddings.npy')
            self.movie_embeddings = np.load('./processed_data/sbert/movie_title_embeddings.npy')
            book_id_map = pd.read_csv('./processed_data/sbert/book_id_map.csv', index_col=0)
            movie_id_map = pd.read_csv('./processed_data/sbert/movie_id_map.csv', index_col=0)
            self.book_id_to_idx = {str(row['book_id']): idx for idx, row in book_id_map.iterrows()}
            self.movie_id_to_idx = {row['movieId']: idx for idx, row in movie_id_map.iterrows()}
            self.idx_to_movie_id = {idx: mid for mid, idx in self.movie_id_to_idx.items()}
            print("-> Complete S-BERT Embedding Loading.")
        except FileNotFoundError:
            print("[Warning] The S-BERT embedding file could not be found, disabling semantic rebalancing.")
            self.book_embeddings = None


    '''
    If personalization recommendation is not possible (lack of datasets, cold start), 
    make a list of the most popular movies in advance and print them out.
    '''
    def _prepare_popularity_fallback(self):
        movie_info = self.movie_data_df[['movieId', 'title_cleaned']].drop_duplicates().set_index('movieId')
        
        #Self.movie_data_df['rating']>= 4 parts can be adjusted to determine how many or more movies should be based on 'good'
        popularity_scores = self.movie_data_df[self.movie_data_df['rating'] >= 4]['movieId'].value_counts()
        self.popular_movies_df = popularity_scores.to_frame('popularity').join(movie_info)
        print("-> Ready for popular Fallback recommendation list.")



    def _find_similar_users_with_vectors(self, target_genre_vector, target_decade_vector, k=20):
        """[Internal Helper] Finds a similar movie user based on the profile vector entered."""
        # Calculate similarity based on genre preferences
        genre_sims = cosine_similarity(target_genre_vector, self.movie_genre_profile_df.values)[0]
        
        # Calculate similarity based on decade preferences
        decade_sims = cosine_similarity(target_decade_vector, self.movie_decade_profile_df.values)[0]
        
        # Combine similarities with a weighted sum to get the final similarity score
        final_sims = (self.genre_weight * genre_sims) + (self.decade_weight * decade_sims)
        
        # Filter for only users with positive similarity
        positive_sim_indices = np.where(final_sims > 0)[0]
        if len(positive_sim_indices) == 0: return None, None
        
        # Select the top-k most similar users (neighbors)
        top_indices = positive_sim_indices[final_sims[positive_sim_indices].argsort()[-k:][::-1]]
        
        neighbor_ids = self.movie_genre_profile_df.index[top_indices]
        neighbor_scores = final_sims[top_indices]
        return neighbor_ids, neighbor_scores

    '''
    Based on the ratings given by the neighbors found, a user-based collaborative filtering is calculated that calculates the "expected rating" for each movie that the recommended user has not yet seen.
    '''
    def _predict_scores_for_neighbors(self, neighbor_ids, neighbor_scores):
        """[Internal Helper] Predicts movie scores based on given neighborhood information."""
        neighbor_ratings = self.movie_data_df[self.movie_data_df['userId'].isin(neighbor_ids)].copy()
        if neighbor_ratings.empty: return None

        # Calculate weighted ratings by multiplying each rating by the neighbor's similarity score
        similarity_map = dict(zip(neighbor_ids, neighbor_scores))
        neighbor_ratings['similarity'] = neighbor_ratings['userId'].map(similarity_map)
        neighbor_ratings['weighted_rating'] = neighbor_ratings['rating'] * neighbor_ratings['similarity']
        
        # Group by movie and calculate the final predicted score (weighted average)
        g = neighbor_ratings.groupby('movieId')
        # Add epsilon to prevent zero division
        movie_scores = g['weighted_rating'].sum() / (g['similarity'].sum() + 1e-8)
        return movie_scores

    '''
    When the ID of a specific book user is input, the user's "genre taste" and "taste of the times" profiles are searched.

    Find the movie user neighbors who have the most similar tastes to this profile
    '''
    def find_similar_movie_users(self, book_user_id, k=20):
        """Existing method: receive book_user_id, query the profile, and find a similar user."""
        if book_user_id not in self.book_genre_profile_df.index: return None, None
        
        # Retrieve the user's pre-computed genre and decade preference vectors
        target_genre_vector = self.book_genre_profile_df.loc[book_user_id].values.reshape(1, -1)
        target_decade_vector = self.book_decade_profile_df.loc[book_user_id].values.reshape(1, -1)
        
        # Use the internal helper to find neighbors
        return self._find_similar_users_with_vectors(target_genre_vector, target_decade_vector, k)

    def recommend_movies(self, book_user_id, k=20, top_n=10):
        """Existing Main Recommendation Function."""
        print(f"\n--- Start movie recommendation process for users of '{book_user_id}' ---")
        
        # 1. Find similar users (Collaborative Filtering part!)
        neighbor_ids, neighbor_scores = self.find_similar_movie_users(book_user_id, k)
        
        # ... (Neighbor information generation logic is the same) ...
        neighbor_info = pd.DataFrame({'NeighborID': neighbor_ids, 'Similarity': neighbor_scores}) if neighbor_ids is not None else None
            
        if neighbor_ids is None or neighbor_ids.empty:
            print("-> No Pseudo-Neighbor. Run Fallback.")
            return self.popular_movies_df.head(top_n), neighbor_info

        # 2. Predict scores based on neighbors (Collaborative Filtering part)
        movie_scores = self._predict_scores_for_neighbors(neighbor_ids, neighbor_scores)
        
        if movie_scores is None:
            print("-> No neighbor rating. Run fallback.")
            return self.popular_movies_df.head(top_n), neighbor_info
        
        # 3. Generate initial candidate list
        movie_info = self.movie_data_df[['movieId', 'title_cleaned', 'genres']].drop_duplicates().set_index('movieId')
        top_candidates = movie_scores.to_frame('predicted_score').join(movie_info).sort_values('predicted_score', ascending=False).head(50)
        
        # 4. Re-rank candidates using S-BERT semantic similarity (Content-based part)
        '''
        Among the top_candidates movies that have already been selected based on genre/neighborhood,
        It serves to upload movies similar to text embeddings of books that users liked.
        '''
        if self.book_embeddings is not None and self.semantic_weight > 0:
            # ... (S-BERT Logic) ...
            user_high_rated_books = self.book_data_df[(self.book_data_df['user_id'] == book_user_id) & (self.book_data_df['rating'] >= 3)]['book_id']
            if not user_high_rated_books.empty:
                book_indices = [self.book_id_to_idx.get(str(bid)) for bid in user_high_rated_books if str(bid) in self.book_id_to_idx]
                if book_indices:
                    # ... (The following S-BERT readjustment logic is the same as before) ...
                    ideal_vector = self.book_embeddings[book_indices].mean(axis=0).reshape(1, -1)
                    candidate_movie_ids = top_candidates.index.intersection(self.movie_id_to_idx.keys())
                    movie_indices = [self.movie_id_to_idx[mid] for mid in candidate_movie_ids]
                    
                    if movie_indices:
                        candidate_embeddings = self.movie_embeddings[movie_indices]
                        semantic_sims = cosine_similarity(ideal_vector, candidate_embeddings)[0]
                        sim_series = pd.Series(semantic_sims, index=candidate_movie_ids)
                        top_candidates['semantic_sim'] = sim_series
                        top_candidates['semantic_sim'] = top_candidates['semantic_sim'].fillna(0)
                        predicted_scores_scaled = top_candidates['predicted_score'] / 5.0 
                        top_candidates['final_score'] = (1 - self.semantic_weight) * predicted_scores_scaled + self.semantic_weight * top_candidates['semantic_sim']
                        top_candidates = top_candidates.sort_values('final_score', ascending=False)
                        
        # 5. Augment results with user profile info and return Top-N
        final_recommendations_df = top_candidates.head(top_n).copy()
        
        # [Improvement] Adding User Profile Information
        try:
            user_profile_vector = self.book_genre_profile_df.loc[book_user_id]
            top_user_genres = user_profile_vector.nlargest(5).index.tolist()
            final_recommendations_df.loc[:, 'user_top_genres'] = [top_user_genres] * len(final_recommendations_df)
        except KeyError:
            final_recommendations_df['user_top_genres'] = [[]] * len(final_recommendations_df)
            
        print("-> Finished generating the final recommendation list.")
        
        # Summary of final column order
        display_cols = ['title_cleaned', 'genres', 'predicted_score', 'semantic_sim', 'final_score', 'user_top_genres']
        available_cols = [col for col in display_cols if col in final_recommendations_df.columns]
        return final_recommendations_df[available_cols], neighbor_info
    
    """
    For evaluation, we generate recommendations by directly inputting 
    a temporary profile vector generated inreal time.
    Complex logics such as S-BERT readjustment are excluded for consistency in the evaluation.
    """
    def recommend_movies_with_temp_profile(self, temp_genre_profile, temp_decade_profile, k=20, top_n=10):
        # 1. Browse Pseudo-Neighbor with the entered temporary profile vector
        target_genre_vector = temp_genre_profile.values.reshape(1, -1)
        target_decade_vector = temp_decade_profile.values.reshape(1, -1)
        
        neighbor_ids, neighbor_scores = self._find_similar_users_with_vectors(target_genre_vector, target_decade_vector, k)
        
        if neighbor_ids is None or neighbor_ids.empty:
            return None # Return None instead of Fallback when evaluating

        # 2. Predicting scores based on neighborhood
        movie_scores = self._predict_scores_for_neighbors(neighbor_ids, neighbor_scores)
        
        if movie_scores is None:
            return None
        
        # 3. Return final recommendation list (purely without additional logic such as S-BERT)
        recommendations_df = movie_scores.to_frame('predicted_score').sort_values('predicted_score', ascending=False)
        
        return recommendations_df.head(top_n)
    
  