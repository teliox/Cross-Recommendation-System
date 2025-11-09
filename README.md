# Cross-Recommendation-System
This project implements a hybrid recommendation system that recommends movies to users based on their book preferences. It leverages user rating patterns and content features (genre, publication year, title semantics) to bridge the gap between two different domains: books and movies.

📌 Table of Contents
Project Overview
System Architecture & Pipeline
Core Features & Filtering Methods
File Structure & Scripts
How to Run
Analysis & Results
📖 Project Overview
The primary business objective of this project is to enhance user engagement by expanding content discovery across different media types. By analyzing a user's taste in books, the system provides personalized movie recommendations, aiming to:
Increase platform retention by introducing users to new content domains.
Solve the cold-start problem for users who are new to the movie domain but have an established history in the book domain.
Create a richer, more holistic user profile by combining preferences from multiple sources.
Datasets
Book Domain: Goodreads Book Datasets (Reviews, Book Metadata, Genre Info)
Movie Domain: MovieLens 1M Dataset
⚙️ System Architecture & Pipeline
The system operates in two main phases: an offline pre-computation phase for building models and profiles, and an online phase for generating recommendations in real-time.
(Note: You can create a diagram using tools like diagrams.net or Lucidchart and replace the URL)
Offline Pipeline:
Initial Preprocessing: Raw data from Goodreads and MovieLens are cleaned, merged, and standardized. Key scripts: create_book_metadata.py, create_final_book_dataset.py, preprocess_movielens_full.py.
Data Splitting: The processed datasets are split into training (80%) and testing (20%) sets based on user IDs. Key script: splitData.py.
Feature Engineering & Profiling: Using the training data only, various user profiles are generated. Key script: run_preprocessing.py.
User-Genre Profiles: Captures a user's affinity for different genres.
User-Decade Profiles: Captures a user's preference for content from different eras.
IDF Calculation: Computes the importance of each genre.
Title Embedding: Book and movie titles are converted into semantic vectors using a pre-trained S-BERT model. Key script: embed_titles.py.
Baseline Model Training: A LightFM model is trained on the combined training data to be used for validation. Key script: lightfm_baseline.py.
Online Pipeline (Recommendation Generation):
A book_user_id is provided as input.
The system finds the most similar movie users (neighbors) by comparing their pre-computed Genre and Decade profiles.
The system predicts scores for movies based on the ratings of these neighbors.
The initial list of candidates is re-ranked using the semantic similarity of their titles (from S-BERT).
The final Top-N movie recommendations are presented to the user.
✨ Core Features & Filtering Methods
Our model is a Hybrid Recommendation System that intelligently combines multiple filtering techniques:
1. Content-Based Filtering
This is the foundation of our system, used to model user tastes and bridge the two domains.
User Profiling: We create user profiles based on content features (genres, year). Each user is represented by a vector indicating their preference for each genre and decade. This happens in profiler.py.
Semantic Re-ranking: We use S-BERT (sentence-transformers/all-MiniLM-L6-v2) to embed the semantic meaning of titles. This allows us to re-rank recommendations based on thematic and atmospheric similarity, which goes beyond simple genre matching. This logic is in recommender.py.
2. Memory-Based Collaborative Filtering (User-Based)
This technique is used to generate the actual recommendations.
Neighborhood Formation: We find similar users by calculating the cosine similarity between the content-based profiles (genre + decade) of the target book user and all movie users. The GENRE_WEIGHT hyperparameter controls the balance between genre and decade similarity. This happens in recommender.py.
Rating Prediction: We predict a user's rating for a movie by calculating the weighted average of ratings given by their most similar neighbors. This also happens in recommender.py.
3. Model-Based Collaborative Filtering (as a Baseline)
LightFM: We use LightFM, a powerful matrix factorization model, as a baseline and a "judge" for our model's performance. It learns latent representations for users and items from both interaction data and feature data. This is implemented in lightfm_baseline.py.
📂 File Structure & Scripts
main.py: Main script for running a single-user recommendation analysis and demo.
run_preprocessing.py: Executes the full preprocessing pipeline to generate user profiles from training data.
run_cross_validation.py: (Or tuning.py) Performs evaluation by comparing our model's recommendations against the LightFM latent space.
splitData.py: Splits the raw datasets into training and testing sets.
lightfm_baseline.py: Trains and saves the LightFM baseline model.
embed_titles.py: Generates and saves S-BERT embeddings for all titles.
preprocessor.py: Contains all functions related to data cleaning, feature extraction (genres, decades), and one-hot encoding.
profiler.py: Contains all functions for creating the various user profile matrices (genre, decade, IDF-weighted).
recommender.py: The core recommendation engine, implemented as the CrossDomainRecommender class.
🚀 How to Run
Setup:
Create a conda environment with Python 3.11.
Install required libraries: pip install pandas numpy scikit-learn tqdm sentence-transformers lightfm
Place the raw datasets in the dataset/ directory.
Execute Preprocessing Pipeline (Run these only once):
Split the data: python splitData.py
Generate title embeddings: python embed_titles.py
Train the LightFM model: python lightfm_baseline.py
Generate user profiles for our model: python run_preprocessing.py
Run Analysis & Recommendation:
To see a detailed analysis for a few random users, run:
code
Bash
python main.py
To perform hyperparameter tuning (Grid Search), run:
code
Bash
python tuning.py
📊 Analysis & Results
Our project involved both qualitative and quantitative analysis.
Qualitative Analysis: We conducted case studies on several user personas (e.g., "Classic Thriller Fan," "Epic Fantasy Fan"). Our model demonstrated a strong ability to capture the user's core genre and decade preferences, providing highly relevant and explainable recommendations. LightFM, in contrast, often provided more diverse and serendipitous recommendations.
Quantitative Analysis: We implemented a novel cross-model validation approach. Instead of using traditional precision/recall, we measured the cosine similarity between our model's recommendations and the user's known book preferences within the latent space learned by LightFM. Hyperparameter tuning via Grid Search revealed that a balanced contribution of genre (GENRE_WEIGHT=0.7) and semantic similarity (SEMANTIC_WEIGHT=0.3) yielded the most plausible results from LightFM's perspective. This demonstrates that our model's recommendations are consistent with those of a state-of-the-art baseline model.
For detailed results and visualizations (such as t-SNE plots of the embedding space), please refer to our final report and presentation slides.
