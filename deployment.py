import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

# Load the CSV file
df = pd.read_csv("data.csv")

# Handling missing values, feature engineering, and data preprocessing...

# Load the K-means clustering model (you need to train it or load a pre-trained one)
# kmeans = load_model("kmeans_model.h5")

# Load the Random Forest popularity prediction model (you need to train it or load a pre-trained one)
# rf_model = load_model("rf_model.h5")

# Define a function to get song recommendations from the same cluster
def get_cluster_recommendations(song_name, kmeans_model, data):
    song_index = data[data['name'] == song_name].index[0]
    cluster_label = kmeans_model.labels_[song_index]
    cluster_songs = data[kmeans_model.labels_ == cluster_label]['name']
    return cluster_songs

# Define a function to get song recommendations based on predicted popularity
def get_popularity_recommendations(song_name, model, data):
    song_features = data[data['name'] == song_name][features]
    predicted_popularity = model.predict(song_features)
    similar_songs = data[data['popularity'] >= predicted_popularity[0]].sort_values(by='popularity', ascending=False)
    return similar_songs['name']

# Define Streamlit app
def main():
    st.title("Music Recommendation System")

    # Add Streamlit widgets and user interface elements
    user_input = st.text_input("Enter Song Name:", "Song Name")

    if st.button("Get Cluster Recommendations"):
        cluster_recommendations = get_cluster_recommendations(user_input, kmeans, df)
        st.header("Cluster Recommendations:")
        st.write(cluster_recommendations)

    if st.button("Get Popularity Recommendations"):
        popularity_recommendations = get_popularity_recommendations(user_input, rf_model, df)
        st.header("Popularity Recommendations:")
        st.write(popularity_recommendations)

if __name__ == "__main__":
    main()
