# **Spotify Music Recommendation System**

This project is a Spotify music recommendation system that uses a K-Nearest Neighbors (KNN) model to recommend songs based on user preferences or existing song features. The model is trained on a preprocessed dataset containing various audio features and metadata of songs

The various libraries used in the model are "pandas", "scikit-learn", "joblib", and "ipywidgets"

The project incorporated 11 numerical features and 2 categorical features and evaluates the pool of songs in the dataset to find 10 songs suitable to the chosen preferences i.e selected features.

numerical_features = ['danceability', 'energy', 'loudness', 'mode', 'speechiness','acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

categorical features = categorical_features = ['track_artist', 'playlist_genre']

The feature weights to the specific numerical features are:
  feature_importance_weights = {
    'danceability': 1.2,
    'energy': 1.0,
    'loudness': 0.8,
    'mode': 1.0,
    'speechiness': 0.9,
    'acousticness': 0.7,
    'instrumentalness': 0.6,
    'liveness': 0.5,
    'valence': 1.1,
    'tempo': 0.9,
    'duration_ms': 0.8
}
and the Categorical weights are a uniform value of 2.0

The user interface to the project looks like this:

![image](https://github.com/user-attachments/assets/39a91619-215a-4fb8-b635-bd494b4ab203)

An Example song recommendation according to custom feature parameters:

![image](https://github.com/user-attachments/assets/45297305-230f-45dc-9015-0468917e57c9)

All recommendations and input parameters are saved to the Output.csv file


