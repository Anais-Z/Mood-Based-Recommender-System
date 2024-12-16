# -*- coding: utf-8 -*-
"""recommender_songs.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MLKzwxDZ4P7AhRQ2wEo_vNxy9WstN6oy
"""

import pandas as pd

# Try using a different delimiter like a semicolon or tab
df = pd.read_csv('theSongs.csv')

# Find rows with null values
null_rows = df[df.isnull().any(axis=1)]

# Load the dataset
df = pd.read_csv('theSongs.csv')





tf = pd.read_csv('MillionDollarSongs.csv')

tf
client_id = 'f54c881a659e44ff9d167a3b8e1d2f59'

client_secret  = '20a5b750b3a545aa81ed84698476d5db'

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv("dataset.csv")

# Convert mood-related columns to numeric and coerce errors to NaN
mood_features = ['danceability', 'energy', 'loudness', 'valence', 'tempo']
for feature in mood_features:
    df[feature] = pd.to_numeric(df[feature], errors='coerce')

# Drop rows with NaN values in mood-related columns
df = df.dropna(subset=mood_features)

# Encode the target (track_genre) if it's categorical
encoder = LabelEncoder()
y = encoder.fit_transform(df['track_genre'])

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(df[mood_features])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(y.unique()), activation='softmax')  # Multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy}")

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv("dataset.csv")

# Convert mood-related columns to numeric and coerce errors to NaN
mood_features = ['danceability', 'energy', 'loudness', 'valence', 'tempo']
for feature in mood_features:
    df[feature] = pd.to_numeric(df[feature], errors='coerce')

# Drop rows with NaN values in mood-related columns
df = df.dropna(subset=mood_features)

# Encode the target (track_genre) if it's categorical
encoder = LabelEncoder()
y = encoder.fit_transform(df['track_genre'])

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(df[mood_features])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')  # Multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
# Load dataset
df = pd.read_csv('dataset.csv')

# Features and target
features = ['danceability', 'energy', 'acousticness', 'loudness', 'valence', 'tempo']
X = df[features]
y = df['track_genre']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# User mood is 'happy'
user_mood = 'happy'

# Filter songs for happy mood based on valence (positive mood), energy (upbeat), and danceability
if user_mood == 'happy':
    happy_songs = df[(df['valence'] > 0.7) & (df['energy'] > 0.7) & (df['danceability'] > 0.7)]  # Filter for upbeat and happy songs

    # Optionally, recommend top 10 songs by popularity
    recommended_songs = happy_songs.nlargest(10, 'popularity')  # Top 10 songs by popularity

    # Output the recommended songs
    print("Top 10 songs recommended for a happy mood:")
    print(recommended_songs[['track_name', 'artists', 'valence', 'energy', 'danceability', 'popularity']])

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('dataset.csv')

# Features and target
features = ['danceability', 'energy', 'acousticness', 'loudness', 'valence', 'tempo']
X = df[features]
y = df['track_genre']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Function to map mood to feature values dynamically
def map_mood_to_features(mood):
    """ Map user mood to features """
    mood_map = {
        'happy': {'valence': 0.9, 'energy': 0.8, 'danceability': 0.85},
        'sad': {'valence': 0.2, 'energy': 0.3, 'danceability': 0.4},
        'energetic': {'valence': 0.7, 'energy': 0.9, 'danceability': 0.95},
        'calm': {'valence': 0.5, 'energy': 0.4, 'danceability': 0.45},
        'angry': {'valence': 0.1, 'energy': 0.9, 'danceability': 0.7},
        # Add more moods as needed
    }

    # Default to a neutral mood if input is not recognized
    return mood_map.get(mood.lower(), {'valence': 0.5, 'energy': 0.5, 'danceability': 0.5})

# Ask user for mood input (e.g., "happy", "sad", etc.)
user_mood = input("How are you feeling today? (e.g., happy, sad, energetic) ")

# Map the user mood to feature values
user_features = map_mood_to_features(user_mood)

# Find songs that match user mood features
# Filter songs whose features are close to the user's input mood features
filtered_songs = df[
    (df['valence'] >= user_features['valence'] - 0.1) &
    (df['energy'] >= user_features['energy'] - 0.1) &
    (df['danceability'] >= user_features['danceability'] - 0.1)
]

# Optionally, recommend top 10 songs by popularity or other criteria
recommended_songs = filtered_songs.nlargest(10, 'popularity')

# Output the recommended songs
print("Top 10 songs recommended for your mood:")
print(recommended_songs[['track_name', 'artists', 'valence', 'energy', 'danceability', 'popularity']])

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('dataset.csv')

# Features and target
features = ['danceability', 'energy', 'acousticness', 'loudness', 'valence', 'tempo']
X = df[features]
y = df['track_genre']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Function to map mood to feature values dynamically
def map_mood_to_features(mood):
    """ Map user mood to features """
    mood_map = {
        'happy': {'valence': 0.9, 'energy': 0.8, 'danceability': 0.85},
        'sad': {'valence': 0.2, 'energy': 0.3, 'danceability': 0.4},
        'energetic': {'valence': 0.7, 'energy': 0.9, 'danceability': 0.95},
        'calm': {'valence': 0.5, 'energy': 0.4, 'danceability': 0.45},
        'angry': {'valence': 0.1, 'energy': 0.9, 'danceability': 0.7},
        # Add more moods as needed
    }

    # Default to a neutral mood if input is not recognized
    return mood_map.get(mood.lower(), {'valence': 0.5, 'energy': 0.5, 'danceability': 0.5})

# Ask user for mood input (e.g., "happy", "sad", etc.)
user_mood = input("How are you feeling today? (e.g., happy, sad, energetic) ")

# Map the user mood to feature values
user_features = map_mood_to_features(user_mood)

# Find songs that match user mood features
# Filter songs whose features are close to the user's input mood features
filtered_songs = df[
    (df['valence'] >= user_features['valence'] - 0.1) &
    (df['energy'] >= user_features['energy'] - 0.1) &
    (df['danceability'] >= user_features['danceability'] - 0.1)
]

# Remove duplicate songs based on track name or track ID
filtered_songs = filtered_songs.drop_duplicates(subset=['track_name'])

# Optionally, recommend top 10 songs by popularity or other criteria
recommended_songs = filtered_songs.nlargest(10, 'popularity')

# Output the recommended songs
print("Top 10 songs recommended for your mood:")
print(recommended_songs[['track_name', 'artists', 'valence', 'energy', 'danceability', 'popularity']])