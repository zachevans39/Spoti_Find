from flask import Flask, render_template, redirect, url_for, request
import os
import json
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from functions.functions import extract_user_playlist, song_chooser, generate_recommendation
from dotenv import load_dotenv

load_dotenv()
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')

client_credentials_manager = SpotifyClientCredentials(
    client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

app = Flask(__name__)


@app.route('/')
def echo():
    return render_template('index.html')


@app.route('/input')
def index():
    return render_template('input.html')


@app.route('/about')
def contacts():
    return render_template('about.html')


@app.route('/recommendations', methods=['POST'])
def recommendations():
    # Interacting with the user on the html side
    URL = request.form['URL']
    number_requested = int(request.form['recommend-size'])

    # Extract the songs from the user input playlist
    user_playlist = extract_user_playlist(URL)
    song = song_chooser(URL)
    recommended_songs = generate_recommendation(song)

    # Shuffle the order of the songs in the list
    if len(recommended_songs)>1:
        recommended_songs = recommended_songs.sample(number_requested)
        print(recommended_songs)
    else:
        recommended_songs

    return render_template('recommendations.html', user_playlist=user_playlist, recommended_songs=recommended_songs, URL=URL, recommend_size=number_requested)


# debugger to edit while running
if __name__ == "__main__":
    app.run(debug=True)
