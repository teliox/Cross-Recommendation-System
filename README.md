# Cross-Recommendation-System
This project implements a hybrid recommendation system that recommends movies to users based on their book preferences. It leverages user rating patterns and content features (genre, publication year, title semantics) to bridge the gap between two different domains: books and movies.

# Cross-Domain Recommendation System: From Books to Movies

This project implements a hybrid recommendation system that recommends movies to users based on their book preferences. It leverages user rating patterns and content features (genre, publication year, title semantics) to bridge the gap between two different domains: books and movies.

## Data Setup

This project requires the following datasets. Please download them from their original sources as redistribution is not permitted.

1.  **MovieLens 1M Dataset**:
    -   Download from the official GroupLens website: [https://grouplens.org/datasets/movielens/1m/](https://grouplens.org/datasets/movielens/1m/)
    -   Unzip the file and place the `ml-1m` folder inside the `dataset/` directory. The final path should be `dataset/ml-1m/`.

2.  **Goodreads Book Graph**:
    -   Download the necessary files from the UCSD project page: [https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home)
    -   Required files:
        -   `goodreads_reviews_dedup.json.gz`
        -   `goodreads_books.json.gz`
        -   `gooreads_book_genres_initial.json.gz`
    -   Create a `goodreads` folder inside the `dataset/` directory and place all three `.json.gz` files inside it. The final path should be `dataset/goodreads/`.

After setting up the data, your `dataset` directory should look like this:

## Table of Contents
- [Project Overview](#-project-overview)
- [System Architecture & Pipeline](#-system-architecture--pipeline)
- [Core Features & Filtering Methods](#-core-features--filtering-methods)
- [File Structure & Scripts](#-file-structure--scripts)
- [How to Run](#-how-to-run)
- [Analysis & Results](#-analysis--results)

## Project Overview

The primary business objective of this project is to enhance user engagement by expanding content discovery across different media types. By analyzing a user's taste in books, the system provides personalized movie recommendations, aiming to:
- **Increase platform retention** by introducing users to new content domains.
- **Solve the cold-start problem** for users who are new to the movie domain but have an established history in the book domain.
- **Create a richer, more holistic user profile** by combining preferences from multiple sources.

### Datasets
- **Book Domain**: [Goodreads Book Datasets](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) (Reviews, Book Metadata, Genre Info)
- **Movie Domain**: [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)

## System Architecture & Pipeline

The system operates in two main phases: an offline pre-computation phase for building models and profiles, and an online phase for generating recommendations in real-time.

 
*(Note: You can create a diagram using tools like diagrams.net or Lucidchart, upload it to a site like Imgur, and replace the URL)*

**Offline Pipeline:**
1.  **Initial Preprocessing**: Raw data from Goodreads and MovieLens are cleaned, merged, and standardized. Key scripts: `create_book_metadata.py`, `create_final_book_dataset.py`, `preprocess_movielens_full.py`.
2.  **Data Splitting**: The processed datasets are split into training (80%) and testing (20%) sets based on user IDs. Key script: `splitData.py`.
3.  **Feature Engineering & Profiling**: Using the **training data only**, various user profiles are generated. Key script: `run_preprocessing.py`.
    -   **User-Genre Profiles**: Captures a user's affinity for different genres.
    -   **User-Decade Profiles**: Captures a user's preference for content from different eras.
    -   **IDF Calculation**: Computes the importance of each genre.
4.  **Title Embedding**: Book and movie titles are converted into semantic vectors using a pre-trained S-BERT model. Key script: `embed_titles.py`.
5.  **Baseline Model Training**: A `LightFM` model is trained on the combined training data to be used for validation. Key script: `lightfm_baseline.py`.

**Online Pipeline (Recommendation Generation):**
1.  A `book_user_id` is provided as input.
2.  The system finds the most similar movie users (neighbors) by comparing their pre-computed Genre and Decade profiles.
3.  The system predicts scores for movies based on the ratings of these neighbors.
4.  The initial list of candidates is re-ranked using the semantic similarity of their titles (from S-BERT).
5.  The final Top-N movie recommendations are presented to the user.

## Core Features & Filtering Methods

Our model is a **Hybrid Recommendation System** that intelligently combines multiple filtering techniques:

#### 1. Content-Based Filtering
This is the foundation of our system, used to model user tastes and bridge the two domains.
-   **User Profiling**: We create user profiles based on content features (`genres`, `year`). Each user is represented by a vector indicating their preference for each genre and decade. This happens in **`profiler.py`**.
-   **Semantic Re-ranking**: We use **S-BERT** (`sentence-transformers/all-MiniLM-L6-v2`) to embed the semantic meaning of titles. This allows us to re-rank recommendations based on thematic and atmospheric similarity, which goes beyond simple genre matching. This logic is in **`recommender.py`**.

#### 2. Memory-Based Collaborative Filtering (User-Based)
This technique is used to generate the actual recommendations.
-   **Neighborhood Formation**: We find similar users by calculating the cosine similarity between the content-based profiles (genre + decade) of the target book user and all movie users. The `GENRE_WEIGHT` hyperparameter controls the balance between genre and decade similarity. This happens in **`recommender.py`**.
-   **Rating Prediction**: We predict a user's rating for a movie by calculating the weighted average of ratings given by their most similar neighbors. This also happens in **`recommender.py`**.

#### 3. Model-Based Collaborative Filtering (as a Baseline)
-   **LightFM**: We use `LightFM`, a powerful matrix factorization model, as a baseline and a "judge" for our model's performance. It learns latent representations for users and items from both interaction data and feature data. This is implemented in **`lightfm_baseline.py`**.

## File Structure & Scripts

CrossRecommendation/
├── main.py # Main script for demo and single-user analysis
├── run_preprocessing.py # Generates all user profiles from training data
├── lightfm_baseline.py # Trains the LightFM baseline model
├── embed_titles.py # Generates S-BERT embeddings for titles
├── splitData.py # Splits data into train/test sets
├── tuning.py # (Optional) Script for hyperparameter grid search
├── preprocessor.py # Module for data cleaning and feature engineering
├── profiler.py # Module for creating user profile matrices
├── recommender.py # The core recommendation engine class
├── dataset/ # Raw data files
└── processed_data/ # All generated files (split data, profiles, embeddings, models)

## How to Run

1.  **Setup**:
    -   Create a conda environment with Python 3.11: `conda create -n ml-project python=3.11`
    -   Activate the environment: `conda activate ml-project`
    -   Install required libraries: `pip install pandas scikit-learn tqdm sentence-transformers lightfm`
    -   Place the raw datasets in the `dataset/` directory.

2.  **Execute Preprocessing Pipeline (Run these only once in order)**:
    -   **Step 1**: Create the initial processed CSVs (run `create_book_metadata.py`, `create_final_book_dataset.py`, `preprocess_movielens_full.py`).
    -   **Step 2**: Split the data: `python splitData.py`
    -   **Step 3**: Generate S-BERT embeddings: `python embed_titles.py`
    -   **Step 4**: Train the LightFM model: `python lightfm_baseline.py`
    -   **Step 5**: Generate user profiles for our model: `python run_preprocessing.py`

3.  **Run Analysis & Recommendation**:
    -   To see a detailed analysis for a few random users, run:
        ```bash
        python main.py
        ```

## Analysis & Results

Our project involved both qualitative and quantitative analysis.

-   **Qualitative Analysis**: We conducted case studies on several user personas (e.g., "Classic Thriller Fan," "Epic Fantasy Fan"). Our model demonstrated a strong ability to capture the user's core genre and decade preferences, providing highly relevant and explainable recommendations. LightFM, in contrast, often provided more diverse and serendipitous recommendations.

-   **Quantitative Analysis**: We implemented a novel cross-model validation approach. Instead of using traditional precision/recall, we measured the cosine similarity between our model's recommendations and the user's known book preferences within the latent space learned by LightFM. Hyperparameter tuning via Grid Search revealed that a balanced contribution of genre (`GENRE_WEIGHT=0.7`) and semantic similarity (`SEMANTIC_WEIGHT=0.3`) yielded the most plausible results from LightFM's perspective. This demonstrates that our model's recommendations are consistent with those of a state-of-the-art baseline model.

For detailed results and visualizations (such as t-SNE plots of the embedding space), please refer to our final report and presentation slides.

## Acknowledgements & Data Source

This project would not have been possible without the following publicly available datasets. We are grateful to the researchers who collected and provided them.

-   **MovieLens 1M Dataset**:
    F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

-   **Goodreads Book Graph**:
    Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", in RecSys'18.
    Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19.
    
