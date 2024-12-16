
# Load datasets
import pandas as pd
from IPython.display import display

track_df = pd.read_csv('track_dataset.csv')
playlist_df = pd.read_csv('playlist_dataset.csv')
user_df = pd.read_csv('user_dataset.csv')

# Remove duplicates by keeping the first occurrence
track_df = track_df.drop_duplicates(subset='track_id', keep='first')

# Convert lists to proper list-string format after loading csv
import ast
user_df['playlists'] = user_df['playlists'].apply(ast.literal_eval)
playlist_df['track_list'] = playlist_df['track_list'].apply(ast.literal_eval)

print('Track_df:\n', track_df.columns)
print('\nPlaylist_df:\n', playlist_df.columns)
print('\nUser_df:\n', user_df.columns)

display(user_df)

"""# Pre-processing --> Stratified train-test split
training data gets 80% of each user's playlists
"""

from sklearn.model_selection import train_test_split
from IPython.display import display

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
        train_row = {
            'user_id': user_id,
            'preferred_genres': user_df[user_df['user_id'] == user_id]['preferred_genres'].values[0],  # Extract value
            'playlists': train['playlist_id'].tolist()
        }
        train_user_df = pd.concat([train_user_df, pd.DataFrame([train_row])], ignore_index=True)

        # Add to test set
        test_row = {
            'user_id': user_id,
            'preferred_genres': user_df[user_df['user_id'] == user_id]['preferred_genres'].values[0],  # Extract value
            'playlists': test['playlist_id'].tolist()
        }
        test_user_df = pd.concat([test_user_df, pd.DataFrame([test_row])], ignore_index=True)



    return train_user_df, test_user_df

train_user_df, test_user_df = split_playlists(user_df, playlist_df)

display(train_user_df.shape)
display(test_user_df.shape)

from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Step 1: Extract unique genres and one-hot encode them
genres = track_df['track_genre'].unique()  # Get unique genres
genre_encoder = OneHotEncoder(sparse_output=False)  # Initialize one-hot encoder
genre_one_hot = genre_encoder.fit_transform(track_df[['track_genre']])  # Fit and transform

# Step 2: Add one-hot encoded genres to numerical features
numerical_columns = [
    'popularity', 'duration_ms', 'danceability', 'energy',
    'key', 'loudness', 'mode', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'
]

# Create a new DataFrame with numerical features and one-hot encoded genre
numerical_features = track_df[numerical_columns].values  # Extract numerical features
track_features_with_genre = np.hstack([numerical_features, genre_one_hot])  # Concatenate features

# Create a dictionary mapping track_id -> full feature vector (numerical + genre)
track_feature_dict = {
    track_id: feature_vector
    for track_id, feature_vector in zip(track_df['track_id'], track_features_with_genre)
}

# Step 3: Convert playlists to feature matrices
def get_feature_matrix(track_list):
    """
    Convert a track_list into a matrix of numerical features with one-hot encoded genres.
    """
    features = []
    for track_id in track_list:
        if track_id in track_feature_dict:  # Ensure track_id exists in the dictionary
            features.append(track_feature_dict[track_id])
    return np.array(features)

# Apply to playlist_df
playlist_features = []
for _, row in playlist_df.iterrows():
    track_list = row['track_list']
    feature_matrix = get_feature_matrix(track_list)
    playlist_features.append(feature_matrix)

# # Verify the output
# for idx, matrix in enumerate(playlist_features):
#     print(f"Feature Matrix for Playlist {idx+1}:")
#     print(matrix)
#     print()

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Parameters
window_size = 3  # Number of tracks in the input sequence

# Function to generate input-output pairs
def generate_pairs(feature_matrix, window_size):
    """
    Generate input-output pairs from a feature matrix.
    Args:
      feature_matrix: NumPy array where rows are track feature vectors.
      window_size: Number of tracks in the input sequence.
    Returns:
      inputs: List of input sequences (each a matrix of track features).
      outputs: List of output vectors (each a track feature vector).
    """
    inputs, outputs = [], []
    for i in range(1, len(feature_matrix)):  # Start at index 1 to have a non-empty input
        input_seq = feature_matrix[:i]  # Input is all tracks up to (but not including) i
        output = feature_matrix[i]  # Output is the i-th track
        inputs.append(input_seq)
        outputs.append(output)
    return inputs, outputs

# Generate input-output pairs for all playlists
all_inputs, all_outputs = [], []
for features in playlist_features:
    inputs, outputs = generate_pairs(features, window_size)
    all_inputs.extend(inputs)
    all_outputs.extend(outputs)

# Pad input sequences to fixed length
padded_inputs = pad_sequences(
    [np.array(x).tolist() for x in all_inputs],  # Convert each sequence to a list of lists
    maxlen=window_size,  # Fixed window size
    padding='post',  # Pad at the end
    dtype='float32'
)

# Convert outputs to NumPy array
all_outputs = np.array(all_outputs)

# Verify the shapes of inputs and outputs
print("Padded Inputs Shape:", padded_inputs.shape)
print("Outputs Shape:", all_outputs.shape)

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
test_size = 0.2  # Use 20% of the data for testing
random_state = 0  # For reproducibility

# Perform the split
X_train, X_test, y_train, y_test = train_test_split(
    padded_inputs, all_outputs, test_size=test_size, random_state=random_state
)

# Verify the shapes of the splits
print("Training Inputs Shape:", X_train.shape)
print("Training Outputs Shape:", y_train.shape)
print("Testing Inputs Shape:", X_test.shape)
print("Testing Outputs Shape:", y_test.shape)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Parameters
window_size = X_train.shape[1]  # Length of the input sequence
number_of_features = X_train.shape[2]  # Number of features per track

# Step 1: Define the input layer
inputs = Input(shape=(window_size, number_of_features), name="Input_Layer")

# Step 2: Add convolutional layers
conv1 = Conv1D(filters=128, kernel_size=3, activation='relu', name="Conv1")(inputs)  # Kernel size 3
conv2 = Conv1D(filters=64, kernel_size=1, activation='relu', name="Conv2")(conv1)  # Kernel size 2

# Step 3: Use Global Max Pooling instead of regular pooling
pool = tf.keras.layers.GlobalMaxPooling1D(name="GlobalMaxPooling")(conv2)


# Step 4: Flatten the features
flat = Flatten(name="Flatten")(pool)

# Step 5: Add fully connected (dense) layers
dense1 = Dense(128, activation='relu', name="Dense1")(flat)
dropout = Dropout(0.5, name="Dropout")(dense1)

# Step 6: Output layer
outputs = Dense(number_of_features, activation='linear', name="Output_Layer")(dropout)


model = Model(inputs, outputs, name="Session_CNN")
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


model.summary()

import matplotlib.pyplot as plt

# Parameters
batch_size = 32  
epochs = 50     

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

# Plot training history
def plot_history(history):
    # Plot training & validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training & validation MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error (MAE)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the function to plot
plot_history(history)

def recommend_tracks(user_session, user_history, model, track_feature_dict, num_recommendations=10):
    """
    Generate track recommendations for a user, limited to their historical tracks.

    Args:
      user_session (list): List of track IDs representing the user's current session.
      user_history (list): List of track IDs in the user's historical track pool.
      model (Model): Trained CNN model.
      track_feature_dict (dict): Dictionary mapping track IDs to feature vectors.
      num_recommendations (int): Number of recommendations to generate (default: 10).

    Returns:
      list: Recommended track IDs.
    """
    # Step 1: Convert session track IDs to feature matrix
    session_features = [track_feature_dict[track_id] for track_id in user_session if track_id in track_feature_dict]

    # Extract window size and number of features
    window_size = model.input_shape[1]  # Extract window size from model input
    number_of_features = model.input_shape[2]  # Extract number of features

    # Manual padding
    if len(session_features) < window_size:
        padding_needed = window_size - len(session_features)
        session_features.extend([np.zeros(number_of_features) for _ in range(padding_needed)])
    elif len(session_features) > window_size:
        session_features = session_features[-window_size:]  # Truncate to match the window size

    # Convert to NumPy array
    session_padded = np.array(session_features)

    # Filter track candidates to user's historical tracks
    user_candidate_tracks = {track_id: track_feature_dict[track_id] for track_id in user_history if track_id in track_feature_dict}

    recommendations = []

    # Step 2: Generate recommendations iteratively
    for _ in range(num_recommendations):
        # Ensure session_padded is properly reshaped
        session_input = np.expand_dims(session_padded, axis=0)  # Shape: (1, window_size, number_of_features)

        # Predict the next track's features
        predicted_features = model.predict(session_input).squeeze()

        # Find the closest matching track in the user's history
        closest_track_id = find_closest_track(predicted_features, user_candidate_tracks)

        if closest_track_id is None:
            break  # If no valid track is found, stop recommending

        # Append the recommended track to the session
        recommendations.append(closest_track_id)
        new_track_features = track_feature_dict[closest_track_id]
        session_features.append(new_track_features)

        # Update session_padded
        if len(session_features) > window_size:
            session_features = session_features[-window_size:]  # Keep only the last `window_size` tracks
        session_padded = np.array(session_features)

    return recommendations

def find_closest_track(predicted_features, candidate_tracks):
    """
    Find the closest matching track in the user's candidate track pool based on predicted features.

    Args:
    predicted_features (np.array): Feature vector predicted by the model.
    candidate_tracks (dict): Dictionary mapping track IDs to feature vectors (user's historical tracks).

    Returns:
    int: The track ID of the closest matching track.
    """
    min_distance = float('inf')
    closest_track_id = None
    for track_id, features in candidate_tracks.items():
      distance = np.linalg.norm(features - predicted_features)  # Euclidean distance
      if distance < min_distance:
          min_distance = distance
          closest_track_id = track_id
    return closest_track_id

"""# Evaluation

1. Iterate through each playlist for each user in the test set
2. Find the subset of tracks to evaluate on (instead of recommending on 100K, which is too hard, the subset is only tracks from that user's training & test set)
3. Generate recommendations specific to that user & mood (except baseline algorithms don't consider mood)
4. Evaluates the recommendationed tracks against the tracks in the test playlist using Precision, recall, cosine similarity


"""

from sklearn.metrics.pairwise import cosine_similarity

def evaluate_recommender_algorthim(train_user_df, test_user_df, playlist_df, track_df, top_n_recommendations=10):
    detailed_results = []

    # Select only numeric feature columns
    feature_columns = track_df.select_dtypes(include=['float64', 'int64']).columns

    c=0

    # Iterate through each user
    for _, test_row in test_user_df.iterrows():
        user_id = test_row['user_id']
        test_playlists = test_row['playlists']

        # Get valid tracks for the user (training and test playlists)
        user_playlists = train_user_df[train_user_df['user_id'] == user_id]['playlists'].values[0]
        valid_tracks = playlist_df[playlist_df['playlist_id'].isin(user_playlists + test_playlists)]['track_list'].sum()

        # Iterate through each test playlist
        for test_playlist_id in test_playlists:
            c+=1
            if c%200 == 0:
              test_playlist = playlist_df[playlist_df['playlist_id'] == test_playlist_id]
              test_tracks = test_playlist['track_list'].values[0]  # Tracks in the playlist
              overall_mood = test_playlist['overall_mood'].values[0]  # Mood of the playlist
              overall_genre = test_playlist['overall_genre'].values[0]  # Genre of the playlist

              # Call the function to get a random playlist
              user_playlist = select_random_playlist(user_id, overall_genre, user_df, playlist_df)

            
              recommendations = recommend_tracks(
                  user_session=user_playlist,
                  user_history=valid_tracks,
                  model=model,
                  track_feature_dict=track_feature_dict,
                  num_recommendations=10
              )

              recommended_tracks = recommendations
              print('RECCS: ', recommended_tracks)
             



              # Calculate precision and recall
              hits = set(recommended_tracks).intersection(set(test_tracks))
              precision = len(hits) / len(recommended_tracks) if len(recommended_tracks) > 0 else 0
              recall = len(hits) / len(test_tracks) if len(test_tracks) > 0 else 0

              # Extract numeric features for cosine similarity
              test_track_features = track_df[track_df['track_id'].isin(test_tracks)][feature_columns].values
              recommended_track_features = track_df[track_df['track_id'].isin(recommended_tracks)][feature_columns].values

              if len(test_track_features) > 0 and len(recommended_track_features) > 0:
                  cosine_sim = cosine_similarity(test_track_features, recommended_track_features).mean()
              else:
                  cosine_sim = 0  
              detailed_results.append({
                  'user_id': user_id,
                  'playlist_id': test_playlist_id,
                  'overall_mood': overall_mood,
                  'precision': precision,
                  'recall': recall,
                  'cosine_similarity': cosine_sim
              })

    detailed_results_df = pd.DataFrame(detailed_results)
    return detailed_results_df


import random

def select_random_playlist(user_id, overall_genre, overall_mood, user_df, playlist_df):
    
    user_playlists = user_df[user_df['user_id'] == user_id]['playlists'].values
    if len(user_playlists) == 0:
        print(f"No playlists found for user_id={user_id}.")
        return None

    user_playlists = user_playlists[0]  # Extract the list of playlist IDs for the user

    # Step 2: Filter playlists by overall_genre
    matching_playlists = playlist_df[
        (playlist_df['playlist_id'].isin(user_playlists)) &
        (playlist_df['overall_genre'] == overall_genre)
    ]

    # Step 3: Filter playlists by overall_mood
    matching_playlists = playlist_df[
        (playlist_df['playlist_id'].isin(user_playlists)) &
        (playlist_df['overall_genre'] == overall_genre)
    ]

    if matching_playlists.empty:
        print(f"No playlists found for user_id={user_id} with overall_genre={overall_genre}.")
        return None

    # Step 3: Randomly select a matching playlist
    selected_playlist = matching_playlists.sample(n=1).iloc[0].to_dict()
    return selected_playlist




# Evaluate the recommender system
detailed_results_df = evaluate_recommender_algorthim(
    train_user_df=train_user_df,
    test_user_df=test_user_df,
    playlist_df=playlist_df,
    track_df=track_df
)

print("Detailed Results:")
print(detailed_results_df.head())


aggregated_metrics_by_mood = (
    detailed_results_df
    .groupby('overall_mood')[['precision', 'recall', 'cosine_similarity']]
    .mean()
)

# Compute overall metrics across all moods
overall_metrics = (
    detailed_results_df[['precision', 'recall', 'cosine_similarity']]
    .mean()
)


print("\nAggregated Metrics by Mood:")
print(aggregated_metrics_by_mood)

print("\nOverall Metrics:")
print(overall_metrics)

