# Load datasets
import pandas as pd
track_df = pd.read_csv('track_dataset.csv')
playlist_df = pd.read_csv('playlist_dataset.csv')
user_df = pd.read_csv('user_dataset.csv')

# Convert lists to proper list-string format after loading csv
import ast
user_df['playlists'] = user_df['playlists'].apply(ast.literal_eval)
playlist_df['track_list'] = playlist_df['track_list'].apply(ast.literal_eval)

print('Track_df:\n', track_df.columns)
print('\nPlaylist_df:\n', playlist_df.columns)
print('\nUser_df:\n', user_df.columns)

# Filter tracks that have at least one interaction
def filter_tracks_with_interactions(playlist_df, track_df):
    # Get all track_ids that are in playlists
    tracks_in_playlists = playlist_df['track_list'].sum()  # Flatten the lists
    tracks_with_interactions = set(tracks_in_playlists)

    # Filter track_df to keep only tracks with interactions
    filtered_track_df = track_df[track_df['track_id'].isin(tracks_with_interactions)]
    return filtered_track_df

# Apply the filter
filtered_track_df = filter_tracks_with_interactions(playlist_df, track_df)
#print(track_df.shape)
#print(filtered_track_df.shape)

from sklearn.model_selection import train_test_split

# Step 1: Split Playlists by User
def split_playlists(user_df, playlist_df):

    train_user_df = pd.DataFrame()
    test_user_df = pd.DataFrame()

    for user_id in user_df['user_id']:
        # Get the list of playlists for the user
        user_playlists_list = user_df[user_df['user_id'] == user_id]['playlists'].values[0]

        # Get all of the playlists from playlist_df
        user_playlists_df = playlist_df[playlist_df['playlist_id'].isin(user_playlists_list)]

        ###### USE THIS ONCE USER_DF GENERATION ALLOWS FOR STRATIFICATION ON GENRE & MOOD
        # # Create a third column to be used for stratification
        # user_playlists_df['mood_genre_stratification'] = playlist_df['overall_genre'] + '_' + playlist_df['overall_mood']
        # # Perform stratified split on combined column
        # train, test = train_test_split(user_playlists_df, test_size=0.2, random_state=1, stratify=user_playlists_df['mood_genre_stratification'])

        train, test = train_test_split(user_playlists_df, test_size=0.2, random_state=1)

        # Add to train set
        train_row = {'user_id':user_id, 'preferred_genres':user_df[user_df['user_id'] == user_id]['preferred_genres'], 'playlists':train['playlist_id'].tolist()}
        train_user_df = pd.concat([train_user_df, pd.DataFrame([train_row])], ignore_index=True)

        # Add to test set
        test_row = {'user_id':user_id, 'preferred_genres':user_df[user_df['user_id'] == user_id]['preferred_genres'], 'playlists':test['playlist_id'].tolist()}
        test_user_df = pd.concat([test_user_df, pd.DataFrame([test_row])], ignore_index=True)


    return train_user_df, test_user_df

train_user_df, test_user_df = split_playlists(user_df, playlist_df)

#print(train_user_df.head())
#print(test_user_df.head())

# Step 2: Create Weighted User-Item Interaction Matrix
def create_weighted_user_item_matrix(train_user_df, playlist_df, track_df):
    """
    Create a weighted user-item interaction matrix with interaction counts.

    Parameters:
    - train_user_df: DataFrame linking users to playlists in the training set.
    - playlist_df: DataFrame with playlist information.
    - track_df: Filtered DataFrame with track information (tracks with at least one interaction).

    Returns:
    - user_item_matrix: DataFrame with users as rows, tracks as columns, and interaction counts as values.
    """
    # Get the list of unique users and tracks
    user_ids = train_user_df['user_id'].unique()
    track_ids = track_df['track_id'].unique()

    # Initialize the user-item matrix with zeros
    user_item_matrix = pd.DataFrame(0, index=user_ids, columns=track_ids)

    # Populate the matrix
    for _, row in train_user_df.iterrows():
        user_id = row['user_id']
        playlists = row['playlists']

        # Get all tracks in the user's playlists
        user_tracks = playlist_df[playlist_df['playlist_id'].isin(playlists)][['track_list', 'playlist_id']]

        # Flatten the track lists and count occurrences
        for _, playlist_row in user_tracks.iterrows():
            track_list = playlist_row['track_list']  # List of tracks in this playlist
            for track_id in track_list:
                if track_id in user_item_matrix.columns:
                    user_item_matrix.loc[user_id, track_id] += 1  # Increment count

    return user_item_matrix

# Create the weighted user-item interaction matrix
weighted_user_item_matrix = create_weighted_user_item_matrix(train_user_df, playlist_df, filtered_track_df)

#print("Weighted User-Item Interaction Matrix:")
#print(weighted_user_item_matrix.head(1))

# Normalize by subtracting the mean interaction for each user
normalized_matrix = weighted_user_item_matrix.subtract(weighted_user_item_matrix.mean(axis=1), axis=0).fillna(0).astype(float)

from sklearn.metrics.pairwise import cosine_similarity

# Calculate user-user similarity
user_similarity = cosine_similarity(normalized_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=normalized_matrix.index, columns=normalized_matrix.index)

# Calculate item-item similarity
item_similarity = cosine_similarity(normalized_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=normalized_matrix.columns, columns=normalized_matrix.columns)

from scipy.sparse.linalg import svds
import numpy as np

# Ensure matrix is dense and valid
dense_matrix = normalized_matrix.values.astype(float)

# Dynamically determine k
k = min(50, dense_matrix.shape[0] - 1, dense_matrix.shape[1] - 1)
if k <= 0:
    raise ValueError(f"Invalid k={k}. Matrix shape is {dense_matrix.shape}. Ensure the matrix has enough dimensions.")

# Perform SVD
U, sigma, Vt = svds(dense_matrix, k=k)
sigma = np.diag(sigma)

# Reconstruct the matrix
predicted_matrix = np.dot(np.dot(U, sigma), Vt)
predicted_matrix_df = pd.DataFrame(predicted_matrix, index=normalized_matrix.index, columns=normalized_matrix.columns)

import ast

# Ensure track_list contains lists
playlist_df['track_list'] = playlist_df['track_list'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

def recommend_tracks(user_id, mood, predicted_matrix_df, track_df, playlist_df):
    # Check if the user_id exists in the matrix
    if user_id not in predicted_matrix_df.index:
        raise ValueError(f"User ID {user_id} not found in predicted_matrix_df.")
    
    # Get the list of tracks matching the mood
    mood_tracks = playlist_df[playlist_df['overall_mood'] == mood]['track_list'].tolist()
    
    # Flatten the list of lists into a single list and get unique track IDs
    mood_tracks = [track for sublist in mood_tracks for track in sublist]
    mood_tracks = list(set(mood_tracks))  # Remove duplicates
    
    if not mood_tracks:
        raise ValueError(f"No tracks found for mood '{mood}'.")
    
    # Get the predicted scores for the user
    user_predictions = predicted_matrix_df.loc[user_id]

    # Filter for tracks in the mood
    filtered_tracks = user_predictions[user_predictions.index.isin(mood_tracks)]

    if filtered_tracks.empty:
        raise ValueError(f"No tracks found for user {user_id} and mood {mood}.")
    
    # Sort by score and return top recommendations
    recommendations = filtered_tracks.sort_values(ascending=False)
    
    # Ensure unique track names, get the top 10 tracks
    unique_recommendations = []
    track_names_seen = set()
    
    for track_id, score in recommendations.items():  # Use .items() instead of .iteritems()
        track_name = track_df.loc[track_df['track_id'] == track_id, 'track_name'].values[0]
        artist = track_df.loc[track_df['track_id'] == track_id, 'artist'].values[0]  # Get artist for each track
        playlist_name = playlist_df.loc[playlist_df['track_list'].apply(lambda x: track_id in x), 'playlist_name'].values[0]  # Get the playlist name
        
        if track_name not in track_names_seen:
            unique_recommendations.append((track_id, track_name, artist, playlist_name, score))
            track_names_seen.add(track_name)
        
        # Stop once we have 10 unique track names
        if len(unique_recommendations) == 10:
            break

    # If fewer than 10 unique tracks, append more based on predicted score
    if len(unique_recommendations) < 10:
        # Sort all filtered tracks by score
        additional_recommendations = recommendations.loc[~recommendations.index.isin([track[0] for track in unique_recommendations])]
        for track_id, score in additional_recommendations.head(10 - len(unique_recommendations)).items():  # Use .items() instead of .iteritems()
            track_name = track_df.loc[track_df['track_id'] == track_id, 'track_name'].values[0]
            artist = track_df.loc[track_df['track_id'] == track_id, 'artist'].values[0]  # Get artist for each track
            playlist_name = playlist_df.loc[playlist_df['track_list'].apply(lambda x: track_id in x), 'playlist_name'].values[0]  # Get the playlist name
            
            if track_name not in track_names_seen:
                unique_recommendations.append((track_id, track_name, artist, playlist_name))
                track_names_seen.add(track_name)
    
    # Convert list to DataFrame and return the top 10 unique track names
    unique_recommendations_df = pd.DataFrame(unique_recommendations, columns=['track_id', 'track_name', 'artist', 'playlist'])
    
    # Return the top 10 recommended tracks with their track names, artists, playlists, and scores
    return unique_recommendations_df[['track_name', 'artist', 'playlist']]

    
# Example usage
user_id = 'user_1'  # Example user ID (ensure this exists in predicted_matrix_df)
mood = 'Mad'  # Example mood (ensure this exists in playlist_df)

recommendations = recommend_tracks(user_id, mood, predicted_matrix_df, track_df, playlist_df)
print(recommendations)
