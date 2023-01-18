import spotipy
import spotipy.util as util
import pandas as pd
import os
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')

client_credentials_manager = SpotifyClientCredentials(
    client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

data = pd.read_csv("data/updated_music.csv", encoding='utf-8')
data.drop(columns='Unnamed: 0',inplace=True)

feature_cols=['acousticness', 'danceability', 'duration_ms', 'energy',
            'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
            'speechiness', 'tempo', 'time_signature', 'valence']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized_df =scaler.fit_transform(data[feature_cols])

# Create a pandas series with song titles as indices and indices as series values 
indices = pd.Series(data.index, index=data['song_name']).drop_duplicates()

# Create cosine similarity matrix based on given matrix
cosine = cosine_similarity(normalized_df)


def extract_user_playlist(url):

    try:
        # Split the url and use Spotipy to retrieve the track information for each song in the playlist
        playlist_url = url.split("/")[4].split("?")[0]
        playlist_tracks = sp.playlist_tracks(playlist_url)

        # For loop to extract the necessary track information
        track_ids = []
        track_titles = []
        track_artist = []
        track_album_art = []
        track_album_url = []

        for track in playlist_tracks['items']:
            track_ids.append(track['track']['id'])
            track_titles.append(track['track']['name'])
            track_album_art.append(track['track']['album']['images'][2]['url'])
            track_album_url.append(track['track']['album']['external_urls']['spotify'])
            artist_list = []
            for artist in track['track']['artists']:
                artist_list.append(artist['name'])
            track_artist.append(artist_list[0])

        # Create a dataframe from the track_ids to bring in features
        features = sp.audio_features(track_ids)

        features_df = pd.DataFrame(data=features, columns=features[0].keys())
        features_df['title'] = track_titles
        features_df['artist'] = track_artist
        features_df['album_art'] = track_album_art
        features_df['album_url'] = track_album_url
        features_df = features_df[['artist','title','album_url','album_art','id','danceability','energy','loudness','speechiness','acousticness','liveness','valence']]
        
        return features_df
    
    except:
        features_df = pd.DataFrame({})
        return features_df

def song_chooser(url):

    try:
        user_df = extract_user_playlist(url)
        clean_df = user_df[['acousticness', 'danceability', 'energy','liveness', 'loudness', 'speechiness', 'valence', 'title']]
        
        user_avg_scores = clean_df.mean(axis=0)
        search_variable = user_avg_scores.idxmax()
        
        distance = []
        for index, row in clean_df.iterrows():
            point1 = user_avg_scores[search_variable]
            point2 = np.array(row[search_variable])
            dist = np.linalg.norm(point1 - point2)
            distance.append(dist)
        
        clean_df['distance'] = distance
        clean_df.set_index('title', inplace= True)
        song_search = clean_df['distance'].idxmin()
        
        return song_search
    
    except:
        return None

def generate_recommendation(song_name, model_type=cosine):
    try:
        """
        Purpose: Function for song recommendations 
        Inputs: song title and type of similarity model
        Output: Pandas series of recommended songs
        """
        # Get song indices
        index=indices[song_name]
        # Get list of songs for given songs
        score=list(enumerate(model_type[index]))
        # Sort the most similar songs
        similarity_score = sorted(score,key = lambda x:x[1],reverse = True)
        # Select the top-20 recommend songs
        similarity_score = similarity_score[1:20]
        top_songs_index = [i[0] for i in similarity_score]
        # Top 10 recommende songs
        top_songs=data['song_name'].iloc[top_songs_index]
        song_list = top_songs.values
        #turn top_songs into a list and get the track uri from main database
        recommended_list = []
        for track in song_list:
            for i, r in data.iterrows():
                if r['uri'] in recommended_list:
                    next
                else:
                    if r['song_name'] == track:
                        recommended_list.append(r['uri'])              

        #using track uri to get front end variables
        track_ids = []
        track_titles = []
        track_artist = []
        track_album_art = []
        track_album_url = []

        for track in recommended_list:
            track = sp.track(track)
            
            track_ids.append(track['id'])
            track_titles.append(track['name'])
            track_album_art.append(track['album']['images'][2]['url'])
            track_album_url.append(track['album']['external_urls']['spotify'])
            artist_list = []
            for artist in track['artists']:
                artist_list.append(artist['name'])
            track_artist.append(artist_list[0])

        #returning as a dataframe
        recommended_df = pd.DataFrame({'id':track_ids, 'title':track_titles, 'artist':track_artist, 'album_art':track_album_art, 'album_url':track_album_url})
        
        return recommended_df

    except:
        print('song is not in our database')
        recommended_df = pd.DataFrame({})
        return recommended_df