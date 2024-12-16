# -*- coding: utf-8 -*-
"""rec_algorithms.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1w7WJd-eLzjyhR-nz3IBiHAramZaZv6xy
"""

import pandas as pd
#look at this
track_df = pd.read_csv("hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv")
playlist_df = pd.read_csv('/content/updated_playlist_dataset.csv')
user_df = pd.read_csv('/content/user_table_with_playlists.csv')

# print('track_df columns: ', track_df.columns)
# print('playlist_df columns: ', playlist_df.columns)
# print('user_df columns :', user_df.columns)

print(track_df['track_name'].head(20))

"""# Split data into 80/20 split for playlists for each user"""

from sklearn.model_selection import train_test_split
import pandas as pd
import ast

# Convert playlists column to Python lists
user_df['playlists'] = user_df['playlists'].apply(ast.literal_eval)
playlist_df['track_list'] = playlist_df['track_list'].apply(ast.literal_eval)

# Step 1: Expand the `playlists` column in `user_df` into individual playlist IDs
user_playlists = user_df.explode('playlists')[['user_id', 'playlists']]
user_playlists.rename(columns={'playlists': 'pid'}, inplace=True)

# Function to split playlists for each user
def train_test_split_user_playlists(user_df, test_size=0.2, random_state=42):
    train_data = []
    test_data = []

    # Group by user_id
    grouped = user_df.groupby('user_id')

    for user, group in grouped:
        # Split playlists for the user
        train, test = train_test_split(group, test_size=test_size, random_state=random_state)

        # Append to train and test lists
        train_data.append(train)
        test_data.append(test)

    # Combine all user splits into final DataFrames
    train_df = pd.concat(train_data).reset_index(drop=True)
    test_df = pd.concat(test_data).reset_index(drop=True)

    return train_df, test_df

# Perform the split
train_df, test_df = train_test_split_user_playlists(user_playlists, test_size=0.2)

print(train_df.head())
print(test_df.head())

# Output the sizes and save the DataFrames
print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# Save to CSV if needed
train_df.to_csv('/content/train_user_playlists.csv', index=False)
test_df.to_csv('/content/test_user_playlists.csv', index=False)

"""# Create the user-track matrix"""

import pandas as pd
import ast

# Step 2: Expand the `track_list` column in `playlist_df` into individual tracks
playlist_tracks = playlist_df.explode('track_list')[['pid', 'track_list']]
playlist_tracks.rename(columns={'track_list': 'track_id'}, inplace=True)

# Step 3: Merge user-to-playlist mapping with playlist-to-track mapping
user_track_mapping_train = train_df.merge(playlist_tracks, on='pid', how='inner')
user_track_mapping_test = test_df.merge(playlist_tracks, on='pid', how='inner')

# Display the resulting minimal user-track dataframe
print(user_track_mapping_train.head())
print(user_track_mapping_test.head())

# Save the user-track dataframe for later use
user_track_mapping_train.to_csv('/content/user_track_mapping_train.csv', index=False)
user_track_mapping_test.to_csv('/content/user_track_mapping_test.csv', index=False)

import pandas as pd


# Load training and test sets
train_df = pd.read_csv('/content/user_track_mapping_train.csv')
test_df = pd.read_csv('/content/user_track_mapping_test.csv')

######
# Popularity-Based Recommendation
######

# Step 1: Calculate track popularity in the training set
track_popularity = (
    train_df.groupby('pid')['track_id']
    .apply(lambda x: x.nunique())  # Count unique users for each track
    .sort_values(ascending=False)
)

# Convert to a DataFrame
track_popularity_df = track_popularity.reset_index()
track_popularity_df.columns = ['track_id', 'popularity']

# Step 2: Get top N popular tracks
N = 10  # Number of tracks to recommend
top_tracks = track_popularity_df.head(N)['track_id'].tolist()




######
# Generating Recommendations for Test Users
######

# Generate recommendations for each user in the test set
recommendations = {
    user: top_tracks for user in test_df['user_id'].unique()
}

# Create a dictionary of test tracks for each user
test_tracks = (
    test_df.groupby('user_id')['track_id']
    .apply(list)
    .to_dict()
)



######
# Evaluation Metrics
######

# Step 3: Calculate Precision, Recall, and Hit Rate
def evaluate_recommendations(recommendations, test_tracks):
    total_precision = 0
    total_recall = 0
    total_hits = 0
    total_users = len(test_tracks)

    for user, test_items in test_tracks.items():
        if user in recommendations:
            rec_items = recommendations[user]
            # True positives: items recommended and in the test set
            true_positives = set(rec_items) & set(test_items)
            precision = len(true_positives) / len(rec_items) if rec_items else 0
            recall = len(true_positives) / len(test_items) if test_items else 0
            hit = 1 if true_positives else 0

            total_precision += precision
            total_recall += recall
            total_hits += hit

    # Average metrics across all users
    avg_precision = total_precision / total_users
    avg_recall = total_recall / total_users
    hit_rate = total_hits / total_users

    return avg_precision, avg_recall, hit_rate

# Evaluate
precision, recall, hit_rate = evaluate_recommendations(recommendations, test_tracks)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Hit Rate: {hit_rate:.4f}")

"""# Collaborative Filtering"""

# 1. Prepare the User-Item Interaction Matrix
import pandas as pd
from scipy.sparse import csr_matrix

# Prepare the training data
user_item_train = train_df.groupby(['user_id', 'track_id']).size().reset_index(name='count')

# Create a user-item interaction matrix
interaction_matrix = user_item_train.pivot(index='user_id', columns='track_id', values='count')
interaction_matrix = interaction_matrix.fillna(0)

# Convert to a sparse matrix for efficient computation
interaction_sparse = csr_matrix(interaction_matrix.values)

# 2. Train a Collaborative Filtering Model
from sklearn.decomposition import TruncatedSVD

# Initialize SVD
n_components = 50  # Number of latent features
svd = TruncatedSVD(n_components=n_components, random_state=42)

# Fit the model to the interaction matrix
user_factors = svd.fit_transform(interaction_sparse)
item_factors = svd.components_.T


# Generate Recommendations
import numpy as np

# Predict scores for all users and items
predicted_scores = np.dot(user_factors, item_factors.T)

# Create a mapping of user IDs and track IDs
user_ids = interaction_matrix.index
track_ids = interaction_matrix.columns

# Convert predictions to a DataFrame for easy handling
predictions_df = pd.DataFrame(predicted_scores, index=user_ids, columns=track_ids)

# Generate top N recommendations for each user
N = 10
recommendations = {
    user: predictions_df.loc[user].nlargest(N).index.tolist() for user in predictions_df.index
}



# Evaluate
# Prepare test data for evaluation
test_tracks = (
    test_df.groupby('user_id')['track_id']
    .apply(list)
    .to_dict()
)

# Evaluate collaborative filtering recommendations
precision, recall, hit_rate = evaluate_recommendations(recommendations, test_tracks)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Hit Rate: {hit_rate:.4f}")

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score
import numpy as np

# Load the data
train_df = pd.read_csv('/content/user_track_mapping_train.csv')
test_df = pd.read_csv('/content/user_track_mapping_test.csv')
track_df = pd.read_csv('hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv')

# Features for similarity computation
feature_columns = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'instrumentalness', 'liveness']

# Normalize the tempo feature
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

track_df[feature_columns] = scaler.fit_transform(track_df[feature_columns])

# Step 1: Create user profiles (average features)
def create_user_profiles(train_df, track_df, feature_columns):
    # Merge training data with track features
    train_with_features = train_df.merge(track_df[['track_id'] + feature_columns], on='track_id', how='inner')

    # Compute average features for each user
    user_profiles = (
        train_with_features.groupby('user_id')[feature_columns]
        .mean()
        .reset_index()
    )
    return user_profiles

user_profiles = create_user_profiles(train_df, track_df, feature_columns)

# Step 2: Compute cosine similarity between user profiles and track features
def recommend_top_n(user_profiles, track_df, feature_columns, train_tracks, N=10):
    track_features = track_df[['track_id'] + feature_columns].set_index('track_id')
    user_profiles_matrix = user_profiles[feature_columns].values
    track_features_matrix = track_features.values
    similarities = cosine_similarity(user_profiles_matrix, track_features_matrix)

    recommendations = {}
    for i, user_id in enumerate(user_profiles['user_id']):
        top_indices = np.argsort(-similarities[i])
        candidate_tracks = track_features.iloc[top_indices].index.tolist()
        # Exclude tracks the user has already interacted with
        unseen_tracks = [t for t in candidate_tracks if t not in train_tracks.get(user_id, set())]
        recommendations[user_id] = unseen_tracks[:N]

    return recommendations

# Prepare train_tracks mapping
train_tracks = train_df.groupby('user_id')['track_id'].apply(set).to_dict()

# Generate recommendations
recommendations = recommend_top_n(user_profiles, track_df, feature_columns, train_tracks, N=10)


# Prepare train_tracks mapping
train_tracks = train_df.groupby('user_id')['track_id'].apply(set).to_dict()

# Generate recommendations
recommendations = recommend_top_n(user_profiles, track_df, feature_columns, train_tracks, N=10)

# Step 3: Evaluate recommendations
from sklearn.metrics import precision_score, recall_score

def evaluate_recommendations(recommendations, test_df, track_df, feature_columns):
    test_tracks = test_df.groupby('user_id')['track_id'].apply(list).to_dict()

    total_precision = 0
    total_recall = 0
    total_hits = 0
    total_users = len(test_tracks)
    total_similarity = 0
    total_users_with_similarity = 0

    for user, test_items in test_tracks.items():
        if user in recommendations:
            rec_items = recommendations[user]
            # Precision/Recall
            true_positives = set(rec_items) & set(test_items)
            precision = len(true_positives) / len(rec_items) if rec_items else 0
            recall = len(true_positives) / len(test_items) if test_items else 0
            total_precision += precision
            total_recall += recall

            # Feature-based cosine similarity
            if len(test_items) > 0 and len(rec_items) > 0:
                test_features = track_df[track_df['track_id'].isin(test_items)][feature_columns].mean()
                rec_features = track_df[track_df['track_id'].isin(rec_items)][feature_columns].mean()
                similarity = cosine_similarity([test_features], [rec_features])[0][0]
                total_similarity += similarity
                total_users_with_similarity += 1

    # Average metrics across all users
    avg_precision = total_precision / total_users
    avg_recall = total_recall / total_users
    avg_similarity = total_similarity / total_users_with_similarity if total_users_with_similarity > 0 else 0

    return avg_precision, avg_recall, avg_similarity

# Evaluate recommendations
precision, recall, avg_similarity = evaluate_recommendations(recommendations, test_df, track_df, feature_columns)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Avg Cosine Similarity: {avg_similarity:.4f}")

# Test overlap between training and test track IDs
train_track_ids = set(train_df['track_id'])
test_track_ids = set(test_df['track_id'])
overlap = train_track_ids & test_track_ids

print(f"Number of overlapping tracks: {len(overlap)}")
if len(overlap) == 0:
    print("No overlap between training and test tracks. Recommendations cannot match test tracks.")

