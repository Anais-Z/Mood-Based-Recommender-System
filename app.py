from flask import Flask, render_template, request
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import numpy as np


app = Flask(__name__)

# Load datasets
track_df = pd.read_csv('track_dataset.csv')
playlist_df = pd.read_csv('playlist_dataset.csv')
user_df = pd.read_csv('user_dataset.csv')

# Convert lists to proper list-string format after loading csv
user_df['playlists'] = user_df['playlists'].apply(ast.literal_eval)
playlist_df['track_list'] = playlist_df['track_list'].apply(ast.literal_eval)

# Filter tracks that have at least one interaction
def filter_tracks_with_interactions(playlist_df, track_df):
    tracks_in_playlists = playlist_df['track_list'].sum()  # Flatten the lists
    tracks_with_interactions = set(tracks_in_playlists)
    filtered_track_df = track_df[track_df['track_id'].isin(tracks_with_interactions)]
    return filtered_track_df

filtered_track_df = filter_tracks_with_interactions(playlist_df, track_df)

# Split Playlists by User
def split_playlists(user_df, playlist_df):
    train_user_df = pd.DataFrame()
    test_user_df = pd.DataFrame()
    for user_id in user_df['user_id']:
        user_playlists_list = user_df[user_df['user_id'] == user_id]['playlists'].values[0]
        user_playlists_df = playlist_df[playlist_df['playlist_id'].isin(user_playlists_list)]
        train, test = train_test_split(user_playlists_df, test_size=0.2, random_state=1)
        train_row = {'user_id':user_id, 'preferred_genres':user_df[user_df['user_id'] == user_id]['preferred_genres'], 'playlists':train['playlist_id'].tolist()}
        train_user_df = pd.concat([train_user_df, pd.DataFrame([train_row])], ignore_index=True)
        test_row = {'user_id':user_id, 'preferred_genres':user_df[user_df['user_id'] == user_id]['preferred_genres'], 'playlists':test['playlist_id'].tolist()}
        test_user_df = pd.concat([test_user_df, pd.DataFrame([test_row])], ignore_index=True)
    return train_user_df, test_user_df

train_user_df, test_user_df = split_playlists(user_df, playlist_df)

# Create Weighted User-Item Interaction Matrix
def create_weighted_user_item_matrix(train_user_df, playlist_df, track_df):
    user_ids = train_user_df['user_id'].unique()
    track_ids = track_df['track_id'].unique()
    user_item_matrix = pd.DataFrame(0, index=user_ids, columns=track_ids)
    for _, row in train_user_df.iterrows():
        user_id = row['user_id']
        playlists = row['playlists']
        user_tracks = playlist_df[playlist_df['playlist_id'].isin(playlists)][['track_list', 'playlist_id']]
        for _, playlist_row in user_tracks.iterrows():
            track_list = playlist_row['track_list']
            for track_id in track_list:
                if track_id in user_item_matrix.columns:
                    user_item_matrix.loc[user_id, track_id] += 1
    return user_item_matrix

weighted_user_item_matrix = create_weighted_user_item_matrix(train_user_df, playlist_df, filtered_track_df)

# Normalize by subtracting the mean interaction for each user
normalized_matrix = weighted_user_item_matrix.subtract(weighted_user_item_matrix.mean(axis=1), axis=0).fillna(0).astype(float)

# Calculate user-user similarity
user_similarity = cosine_similarity(normalized_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=normalized_matrix.index, columns=normalized_matrix.index)

# Calculate item-item similarity
item_similarity = cosine_similarity(normalized_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=normalized_matrix.columns, columns=normalized_matrix.columns)

# SVD for recommendations
dense_matrix = normalized_matrix.values.astype(float)
k = min(50, dense_matrix.shape[0] - 1, dense_matrix.shape[1] - 1)
U, sigma, Vt = svds(dense_matrix, k=k)
sigma = np.diag(sigma)
predicted_matrix = np.dot(np.dot(U, sigma), Vt)
predicted_matrix_df = pd.DataFrame(predicted_matrix, index=normalized_matrix.index, columns=normalized_matrix.columns)

# Function to recommend tracks
def recommend_tracks(user_id, mood, predicted_matrix_df, track_df, playlist_df):
    # Check if the user_id exists in the matrix
    if user_id not in predicted_matrix_df.index:
        raise ValueError(f"User ID {user_id} not found in predicted_matrix_df.")
    
    # Get the list of tracks matching the mood
    mood_tracks = playlist_df[playlist_df['overall_mood'] == mood]['track_list'].tolist()
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
    
    for track_id, score in recommendations.items():
        track_name = track_df.loc[track_df['track_id'] == track_id, 'track_name'].values[0]
        
        # Get all artists for this track (in case of multiple artists)
        artists = track_df.loc[track_df['track_id'] == track_id, 'artists'].values.tolist()
        
        # Get playlist name (assuming one playlist per track, but you can adjust if there are multiple)
       # playlist_name = playlist_df.loc[playlist_df['track_list'].apply(lambda x: track_id in x), 'album_name'].values[0]
        
        if track_name not in track_names_seen:
            unique_recommendations.append({
                'track_name': track_name,
                'artists': artists,          
            })
            track_names_seen.add(track_name)
        
        # Stop once we have 10 unique track names
        if len(unique_recommendations) == 10:
            break

    # If fewer than 10 unique tracks, append more based on predicted score
    if len(unique_recommendations) < 10:
        additional_recommendations = recommendations.loc[~recommendations.index.isin([track['track_name'] for track in unique_recommendations])]
        for track_id, score in additional_recommendations.head(10 - len(unique_recommendations)).items():
            track_name = track_df.loc[track_df['track_id'] == track_id, 'track_name'].values[0]
            
            artists = track_df.loc[track_df['track_id'] == track_id, 'artists'].values.tolist()
            
           # playlist_name = playlist_df.loc[playlist_df['track_list'].apply(lambda x: track_id in x), 'album_name'].values[0]
            
            if track_name not in track_names_seen:
                unique_recommendations.append({
                    'track_name': track_name,
                    'artists': artists,          
                
                })

                track_names_seen.add(track_name)
    
    return unique_recommendations


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        mood = request.form["mood"]
        try:
            recommendations = recommend_tracks("user_1", mood, predicted_matrix_df, track_df, playlist_df)
        except ValueError as e:
            recommendations = str(e)  # Return the error as a string if something goes wrong
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
