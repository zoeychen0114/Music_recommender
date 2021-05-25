# -*- coding: utf-8 -*-

from flask import Flask, request, render_template
import pandas as pd
from Wide_and_deep import run_WD
from tensorflow import keras

app = Flask(__name__)
import os


# Get a full library of tracks
# track_data = pd.read_csv('Track_Library.csv')
# testing = pd.read_csv('testing_full.csv')
loaded_model = keras.models.load_model('wide_and_deep_model_small')
# load data
data_raw = pd.read_csv('data_sample.csv', encoding="utf-8")
track_data = data_raw[['artist_name','track_name','track_uri']]


# Recommendation playlist
def find_playlist(history, seed_uri, loaded_model, data_raw):
    # Call dictionary 
    # row = testing.loc[testing['input']==seed_uri].drop_duplicates('input').reset_index(drop=True)
    # posts1= row.iloc[:,[i for i in list(range(1,41)) if i % 2 != 0]].transpose().reset_index(drop=True)
    # posts2 = row.iloc[:,[i for i in list(range(1,41)) if i % 2 == 0]].transpose().reset_index(drop=True)
    # posts = pd.concat([posts1, posts2], axis = 1)
    # posts.columns = ['track_uri', 'ratings']
    # posts = pd.merge(posts, track_data, how='left', on='track_uri').reset_index(drop=True).drop_duplicates('track_uri')
    
    # Prediction using KNN + W&D model
    posts = run_WD(seed_uri, 20, loaded_model, data_raw)
    posts = pd.concat([posts, history], axis=0).drop_duplicates('track_uri').sort_values('ratings',ascending=False).reset_index(drop=True)
    posts.to_csv('posts.csv', index=None)
    return posts


@app.route('/')

@app.route('/home',methods=['GET', 'POST'])
def home():
    try:
        os.remove('posts.csv')
    except:
        pass
    try:
        os.remove('dislike.csv')
    except:
        pass
    return render_template('home.html')

@app.route('/about',methods=['GET', 'POST'])
def about():

    return render_template('about.html')

@app.route('/algorithm',methods=['GET', 'POST'])
def algorithm():

    return render_template('algorithm.html')

@app.route('/datasource',methods=['GET', 'POST'])
def datasource():

    return render_template('datasource.html')

@app.route('/output',methods=['GET', 'POST'])
def output():

    # Initialization
    artistin = ""
    titlein = ""
    liked = ""
    dislike = ""
    play_idx = ""
    

    # Create empty profiles for current user
    history = pd.DataFrame(columns = ['track_uri', 'ratings', 'artist_name','track_name'])
    dislike_df = pd.DataFrame(columns = ['track_uri'])

    # Check request
    try:
        history = pd.read_csv('posts.csv')
    except:
        pass
    try:
        liked = request.form['liked']
    except:
        pass
    try:
        dislike = request.form['dislike']
    except:
        pass
    try:
        play_idx = request.form['play_idx']
    except:
        pass
    try:
        artistin = request.form['artistin'].upper() # no case sensitive 
        titlein = request.form['titlein'].upper()  # no case sensitive 
    except:
        pass
    try:
        dislike_df =  pd.read_csv('Dislike.csv')
    except:
        pass
    

    # If change current_play_music
    if request.method == "POST":
        if request.form.get("play_idx"):
            nowplay = play_idx
            posts =  history
            track_name = posts.loc[posts['track_uri'] == play_idx, 'track_name'].iloc[0]
            posts.to_csv('posts.csv', index=None)
            return render_template('output.html', posts = history, nowplay=nowplay, track_name=track_name)
        

        # First round recommendation
        elif request.form.get("artistin"): 
            seed = track_data[track_data['artist_name'].str.upper()==artistin] # no case sensitive 
            seed = seed[seed['track_name'].str.upper()==titlein]# no case sensitive
            seed_uri = seed.iloc[0,-1]
            posts = find_playlist(history, seed_uri, loaded_model, data_raw)
            # Seed track to be ranked at top
            seed_df = pd.DataFrame({'track_uri': [seed_uri], 'ratings':[posts['ratings'].iloc[0]], 'artist_name':[artistin],'track_name':[titlein]})
            posts = pd.concat([seed_df, posts], axis=0).drop_duplicates('track_uri').sort_values('ratings',ascending=False).reset_index(drop=True)
            nowplay = posts['track_uri'].iloc[0] # first song as default music to play
            track_name = posts['track_name'].iloc[0]
            # Update users profile
            posts.to_csv('posts.csv', index=None)
            return render_template('output.html', posts = posts, nowplay=nowplay, track_name=track_name)
    
        # If liked is submitted
        elif request.form.get("liked"):
            seed_uri = liked
            posts = find_playlist(history, seed_uri, loaded_model, data_raw)
            nowplay = posts['track_uri'].loc[0]
            #posts.to_csv('posts.csv', index=None)
            track_name = posts['track_name'].iloc[0]
            return render_template('output.html', posts = posts, nowplay=nowplay, track_name=track_name)
    
    # If dislike is submitted
        elif request.form.get("dislike"):
            dislike_df= dislike_df.append({'track_uri': dislike}, ignore_index=True) # update dislike profile
            posts = history[~history.track_uri.isin(dislike_df.track_uri)].sort_values('ratings',ascending=False).reset_index(drop=True)
            dislike_df.to_csv('dislike.csv', index=None)
            nowplay = posts['track_uri'].iloc[0]
            posts.to_csv('posts.csv', index=None)
            track_name = posts['track_name'].iloc[0]
            return render_template('output.html', posts = posts, nowplay=nowplay, track_name=track_name)




if __name__ == '__main__':
    app.run(debug=True)
    
