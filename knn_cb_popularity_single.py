# -*- coding: utf-8 -*-
"""
KNN Content-Based Recommendation
"""

# data science imports
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import sklearn.preprocessing as skpp

    # Scale and normalize data
def scale_norm(data_numerical):
    # Remove row if there's NA values and convert to numpy array
    data_knn = data_numerical.dropna(axis = 0, how = 'any')
    data_knn = np.asarray(data_knn)
    # scale data
    data_knn = skpp.scale(data_knn, axis = 0)
    # normalize data
    stdA = np.std(data_knn,axis = 0)
    stdA = skpp.normalize(stdA.reshape(1,-1)) # the normalize is different from MATLAB's
    
    data_knn_scale = data_knn @ np.diag(np.ones(stdA.shape[1])/stdA[0])
    
    # extract attributes from raw data
    m,n = data_knn_scale.shape
    # print('size of the data:',m,n)
    return data_knn_scale

    # Knn Recommendation
def knn_rank(model_knn, seed_track, data_id, data_knn_scale, num_neighbor):
    
    distances, indices = model_knn.kneighbors(seed_track, n_neighbors = num_neighbor + 1)

    # get list of raw idx of recommendations
    raw_recommends = data_id.iloc[indices[0][1:],]
    result = raw_recommends
    
    return result, distances, indices

def split_numerical_id(track_data):
    # Extract the numerical features only, for KNN analysis
    data_numerical = track_data[['acousticness','danceability','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','valence','time_signature']]
    
    # Extract the identidy data only, for outputs
    data_id = track_data[['album_name','album_uri','artist_name','artist_uri','track_name','track_uri']]

    return data_numerical, data_id


def get_knn_recommend(model_knn, seed_id, data_id, data_knn_scale, num_neighbor):
    
    # Predict using KNN
    seed_num = data_id[data_id.track_uri == seed_id].index[0]
    seed_vector = data_knn_scale[seed_num].reshape(1, -1)
    
    # Predict recommendation using KNN
    result, distance, indices = knn_rank(model_knn, seed_vector, data_id, data_knn_scale, num_neighbor)
    
    return result, distance


def knn_run(seed_id, num_neighbor):
    
    # Read Data and drop duplicated track_uris
    total_data = pd.read_csv('data_sample.csv').drop_duplicates('track_uri').reset_index(drop = True)
    
    # split the input data into numerical features and identity features
    data_numerical, data_id = split_numerical_id(total_data)
    # scale and normalize the data
    data_knn_scale = scale_norm(data_numerical)
    
    data_id = data_id.reset_index(drop = True)
    
    
    # print(data_knn_scale.shape)
    # print(data_id.shape)
    print('Input Seed:', data_id[data_id['track_uri']==seed_id][['artist_name', 'track_name']])
    
    # Model Training
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=num_neighbor, n_jobs=-1)
    model_knn.fit(data_knn_scale)
    
    # Model Predicting
    result, distance = get_knn_recommend(model_knn, seed_id, data_id, data_knn_scale, num_neighbor)
    
    result = result.reset_index(drop=True)
    score = pd.DataFrame(1 - distance).T[1:].reset_index(drop=True)
    
    output = pd.concat([result, score], axis =1).rename(columns={0:'score'})
    
    return output

if __name__ == '__main__':

    # Sample seed track_uri    
    seed_id = '6dr6QeqH62tYUiPezRbinq'
    
    # Specify the number of neighbors for output
    num_neighbor = 100
    
    output = knn_run(seed_id, num_neighbor)

