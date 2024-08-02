import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data
@st.cache
def load_data():
    df = pd.read_csv('/content/modified_songs.csv')
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    df_encoded = pd.get_dummies(df, columns=categorical_features)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df_encoded)
    
    knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
    knn.fit(X)
    
    return df, df_encoded, scaler, knn

# Function to create a user profile
def create_user_profile(preferences, df_encoded):
    user_profile = np.zeros(len(df_encoded.columns))
    for feature, value in preferences.items():
        if feature in df_encoded.columns:
            feature_index = df_encoded.columns.get_loc(feature)
            user_profile[feature_index] = value
    return user_profile

# Load data
df, df_encoded, scaler, knn = load_data()

# Streamlit app
st.title('Song Recommendation App')

st.sidebar.header('Enter Your Song Preferences')

# Input fields for song preferences
danceability = st.sidebar.slider('Danceability', 0.0, 1.0, 0.5)
energy = st.sidebar.slider('Energy', 0.0, 1.0, 0.5)
loudness = st.sidebar.slider('Loudness', -60.0, 0.0, -5.0)
mode = st.sidebar.slider('Mode', 0, 1, 1)
speechiness = st.sidebar.slider('Speechiness', 0.0, 1.0, 0.1)
acousticness = st.sidebar.slider('Acousticness', 0.0, 1.0, 0.1)
instrumentalness = st.sidebar.slider('Instrumentalness', 0.0, 1.0, 0.0)
liveness = st.sidebar.slider('Liveness', 0.0, 1.0, 0.1)
valence = st.sidebar.slider('Valence', 0.0, 1.0, 0.5)
tempo = st.sidebar.slider('Tempo', 50, 200, 120)
duration = st.sidebar.slider('Duration', 30, 300, 180)

# Create a dictionary of user preferences
user_preferences = {
    'danceability': danceability,
    'energy': energy,
    'loudness': loudness,
    'mode': mode,
    'speechiness': speechiness,
    'acousticness': acousticness,
    'instrumentalness': instrumentalness,
    'liveness': liveness,
    'valence': valence,
    'tempo': tempo,
    'duration': duration
}

# Generate user profile
user_profile = create_user_profile(user_preferences, df_encoded)

# Standardize user profile
user_profile_scaled = scaler.transform([user_profile])

# Get song recommendations
distances, indices = knn.kneighbors(user_profile_scaled, n_neighbors=5)
recommendations = df.iloc[indices[0]]

# Display recommendations
st.header('Recommended Songs')
st.write(recommendations[['artist', 'title']])
