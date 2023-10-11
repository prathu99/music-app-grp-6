#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install streamlit')
get_ipython().system('pip install joblib  # For Pickle')
import streamlit as st
import joblib


# In[ ]:


joblib.dump(kmeans, 'kmeans_model.pkl')
kmeans = joblib.load('kmeans_model.pkl')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'music_app.py', 'import streamlit as st\nfrom sklearn.cluster import KMeans\nimport pandas as pd\nimport numpy as np\ndf=pd.read_csv("data.csv")\ndf["duration"]=df["duration_ms"].apply(lambda x: round(x/1000))\ndf.drop("duration_ms", inplace=True, axis=1)\n# Feature selection\nfeatures = [\'acousticness\', \'danceability\', \'energy\', \'instrumentalness\', \'loudness\', \'speechiness\', \'tempo\', \'duration\']\n\n# Create a new DataFrame with only the selected features\nsong_features = df[features]\n\n# Normalize the features (optional, but recommended)\nsong_features_normalized = (song_features - song_features.mean()) / (song_features.max() - song_features.min())\ntotal_dr = df[\'duration\']  # Assuming \'duration\' is a column in your DataFrame\nyears = df[\'year\']\nkmeans = KMeans(n_clusters=10, random_state=42)\n\nkmeans.fit(song_features_normalized)\n\n# Streamlit app\nst.title("Song Recommendation System")\n\n# Input for song name\nsong_name = st.text_input("Enter a song name:")\n\n# Button to trigger recommendations\nif st.button("Get Recommendations"):\n    # Function to get recommendations\n    def get_recommendations(song_name, kmeans_model=kmeans, data=df):\n        song_index = df[df[\'name\'] == song_name].index[0]\n        cluster_label = kmeans_model.labels_[song_index]\n        cluster_songs = df[kmeans_model.labels_ == cluster_label][\'name\']\n        return cluster_songs\n\n    # Get recommendations for the input song\n    recommended_songs = get_recommendations(song_name)\n\n    # Display recommendations\n    if len(recommended_songs) > 0:\n        st.subheader("Recommended Songs:")\n        st.write(recommended_songs)\n    else:\n        st.write("No recommendations found for this song.")')


# In[ ]:





# for random forest

# In[ ]:


get_ipython().run_cell_magic('writefile', 'music_app2.py', 'import streamlit as st\n\nimport pandas as pd\nimport numpy as np\ndf=pd.read_csv("data.csv")\ndf["duration"]=df["duration_ms"].apply(lambda x: round(x/1000))\ndf.drop("duration_ms", inplace=True, axis=1)\n# Feature selection\nfeatures = [\'acousticness\', \'danceability\', \'energy\', \'instrumentalness\', \'loudness\', \'speechiness\', \'tempo\', \'duration\']\n\n# Create a new DataFrame with only the selected features\nX = df[features]\ny = df[\'popularity\']\nfrom sklearn.ensemble import RandomForestRegressor\n# Train a Random Forest model\nrf_model = RandomForestRegressor(n_estimators=100, random_state=42)\nrf_model.fit(X, y)\n# Streamlit app\nst.title("Song Recommendation System")\n\n# Input for song name\nsong_name = st.text_input("Enter a song name:")\n\n# Button to trigger recommendations\nif st.button("Get Recommendations"):\n    # Function to get recommendations\n    def get_recommendations(song_name, model=rf_model, data=df):\n        song_features = data[data[\'name\'] == song_name][features]\n        if not song_features.empty:\n            predicted_popularity = model.predict(song_features)\n            similar_songs = data[data[\'popularity\'] >= predicted_popularity[0]].sort_values(by=\'popularity\', ascending=False)\n            return similar_songs[\'name\']\n        else:\n            return []\n\n    # Get recommendations for the input song\n    recommended_songs = get_recommendations(song_name)\n\n    # Display recommendations\n    if recommended_songs:\n        st.subheader("Recommended Songs:")\n        st.write(recommended_songs)\n    else:\n        st.write("No recommendations found for this song.")')

