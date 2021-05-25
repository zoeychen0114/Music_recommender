# -*- coding: utf-8 -*-
"""
KNN Content-Based Recommendation

By TEAM 151 of OMSA DVA 2021 SPRING

USAGE:
    Out put a list of recommended tracks by calling function:
    
        output = knn_run(seed_id, num_neighbor)
        
        Args:
            seed_id: the Spotify unique track ID of one seed track, e.g.: '6dr6QeqH62tYUiPezRbinq'.
            num_neighbor: the number of recommended tracks for output.
            
DATA:
    This script is based on the dataset of 'data_sample.csv'.
    
"""

# data science imports
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import sklearn.preprocessing as skpp
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

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
    seed_num = data_id[data_id == seed_id].index[0]
    # seed_num = 0
    seed_vector = data_knn_scale[seed_num].reshape(1, -1)
    
    # Predict recommendation using KNN
    result, distance, indices = knn_rank(model_knn, seed_vector, data_id, data_knn_scale, num_neighbor)
    
    return result, distance


def knn_run(seed_id, num_neighbor, data_raw, audio_weight, pid_weight, artist_weight):
    
    # scale the data
    data_knn_scale = data_raw.iloc[:,:-1]
    scale = MinMaxScaler()
    scale.fit(data_knn_scale)
    data_knn_scale = scale.transform(data_knn_scale)
    
    # PCA
    data_knn_PCA = data_knn_scale[:,:9]
    pca = PCA(n_components=2)
    pca.fit(data_knn_PCA)
    x_pca = pca.transform(data_knn_PCA)
    
    data_X = np.hstack([x_pca, data_knn_scale[:,9:]])
    
    # Feature Weighting
    data_X[:,:2] = data_X[:,:2] * audio_weight
    data_X[:,3] = data_X[:,3] * artist_weight
    data_X[:,4] = data_X[:,4] * pid_weight


    data_id = data_raw.iloc[:,-1].reset_index(drop = True)
    
    # print(data_knn_scale.shape)
    # print(data_id.shape)
    # print('Input Seed:', data_id[data_id['track_uri']==seed_id][['artist_name', 'track_name']])
    
    # Model Training
    model_knn = NearestNeighbors(metric='euclidean', algorithm='brute', n_neighbors=num_neighbor, n_jobs=-1)
    model_knn.fit(data_X)
    
    # Model Predicting
    result, distance = get_knn_recommend(model_knn, seed_id, data_id, data_X, num_neighbor)
    
    result = result.reset_index(drop=True)
    score = pd.DataFrame(1 - distance).T[1:].reset_index(drop=True)
    
    output = pd.concat([result, score], axis =1).rename(columns={0:'score'})
    
    return output

if __name__ == '__main__':

    # Sample seed track_uri    
    seed_id = '6dr6QeqH62tYUiPezRbinq' # in pid = 1663
    
    # Specify the number of neighbors for output
    num_neighbor = 1000
    
    data_raw = pd.read_csv('embeddings_uri_final.csv')#.drop_duplicates('14').reset_index(drop = True)

    output = knn_run(seed_id, num_neighbor, data_raw, audio_weight = 1, pid_weight = 1, artist_weight = 1000)

    output.to_csv('output_artist.csv')
