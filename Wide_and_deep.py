#!/usr/bin/env python
# coding: utf-8

from tensorflow import keras
import pandas as pd
import numpy as np
import sklearn.preprocessing as skpp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
pd.set_option('chained_assignment', None)


def get_negative_sampling(data):
    
    result = data.copy()
    pid_length = len(result.pid.unique())
    pid_values = list(range(1, pid_length)) + [0]
    iterations = len(result) // pid_length
    module = len(result) % (pid_length)
    i = 0
    while i < iterations:
        piece = result.iloc[i*pid_length:(i+1)*pid_length] 
        piece.loc[:, 'pid'] = pid_values
        i += 1
        
    result.loc[i*pid_length:(i+1)*pid_length, 'pid'] = pid_values[:module]    
        
    return result


def denseFeature(feat):
    return {'feat': feat}


def sparseFeature(feat, feat_num, embed_dim=4):
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def build_model_columns(data_df, embed_dim=8, test_size=0.2):
    
    # 区分稠密稀疏特征
    sparse_features = ["album_uri", "artist_uri", "pid", "key", "track_uri"]  # categorical，track_uri 加入 pid是label 分开处理
    dense_features = ["acousticness", "danceability", 'energy', 'instrumentalness', 'liveness', 'loudness',
                      'speechiness', 'tempo', 'valence']  # continuous

    # 对稀疏特征做labelEncoder
    for feat in sparse_features:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])

    # 特征工程
    mms = MinMaxScaler(feature_range=(0, 1))
    data_df[dense_features] = mms.fit_transform(data_df[dense_features])

    feature_columns = [[denseFeature(feat) for feat in dense_features]] +                       [[sparseFeature(feat, len(data_df[feat].unique()), embed_dim=embed_dim)
                        for feat in sparse_features]]
    
    # 将sample数据分为train and test （0.8， 0.2）并修改数据格式
    train, test = train_test_split(data_df, test_size=test_size)

    train_x = [train[dense_features].values.astype('float32'), train[sparse_features].values.astype('int32')]
    train_y = train['rating'].values.astype('int32')

    test_x = [test[dense_features].values.astype('float32'), test[sparse_features].values.astype('int32')]
    test_y = test['rating'].values.astype('int32')
    view_test_x = pd.DataFrame(test_x)
    
    return feature_columns, (train_x, train_y), (test_x, test_y)


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


def knn_run(seed_id, num_neighbor, total_tracks):
    
    # Read Data and drop duplicated track_uris
    # split the input data into numerical features and identity features
    data_numerical, data_id = split_numerical_id(total_tracks)
    # scale and normalize the data
    data_knn_scale = scale_norm(data_numerical)
    data_id = data_id.reset_index(drop = True)
    
    # Model Training
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=num_neighbor, n_jobs=-1)
    model_knn.fit(data_knn_scale)
    
    # Model Predicting
    result, distance = get_knn_recommend(model_knn, seed_id, data_id, data_knn_scale, num_neighbor)
    
    return result['track_uri'].tolist() #output


def get_prediction(seed, num, sparse_features, dense_features, total_tracks, data_wd, model_wd):
    
    candidates = knn_run(seed, 1000, total_tracks)
    prediction_data = data_wd[data_wd.id.isin(candidates)]
    predict_x = [prediction_data[dense_features].values.astype('float32'), prediction_data[sparse_features].values.astype('int32')]
    predictions = model_wd.predict(predict_x)
    prediction_data['Prob'] = predictions
    result = prediction_data.sort_values('Prob', ascending=False)[['artist_name', 'track_name','id', 'Prob']].drop_duplicates().reset_index(drop=True).iloc[:num]
    result.columns = ['artist_name', 'track_name','track_uri', 'ratings']
    
    return result

def run_WD(seed, num, model_wd, data_raw):
    

    # transform data for knn prediction
    total_tracks = data_raw.drop_duplicates('track_uri').reset_index(drop = True)
    # transform data for wd prediction
    data_negative_sampled = get_negative_sampling(data_raw)
    data_raw['rating'] = 1
    data_negative_sampled['rating'] = 0
    data_wd = pd.concat([data_raw, data_negative_sampled], axis=0, sort=False)
    feature_columns, train, test = build_model_columns(data_df = data_wd, embed_dim=8, test_size=0.2)
    
    # select data features for wd prediction
    column_selector = ['album_uri', 'artist_uri', 'pid', 'track_uri', 'acousticness', 'danceability', 'energy', 'instrumentalness',
                             'key', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
    sparse_features = ["album_uri", "artist_uri", "pid", "key", "track_uri"]  # categorical，track_uri 加入 pid是label 分开处理
    dense_features = ["acousticness", "danceability", 'energy', 'instrumentalness', 'liveness', 'loudness',
                          'speechiness', 'tempo', 'valence']  # continuous
    # get tracks
    # tracks = data_wd.id.unique()
    
    # get recommendation
    output = get_prediction(seed, num, sparse_features, dense_features, total_tracks, data_wd, model_wd)
    
    return output

if __name__ == '__main__':

    # Sample seed track_uri    
    seed_id = '6dr6QeqH62tYUiPezRbinq'
    num = 20
    # load model
    model_wd = keras.models.load_model('wide_and_deep_model_small')
    # load data
    data_raw = pd.read_csv('data_sample.csv')
    
    output = run_WD(seed_id, num, model_wd, data_raw)