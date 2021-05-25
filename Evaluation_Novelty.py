'''
EVALUATION: Novelty Using popularity of tracks:
    
    REFERENCE: 
        Zhang Y, Séaghdha D, Quercia D, Jambor T (2012) Auralist: introducing serendipity into music recommendation. In: WSDM’12. ACM, New York, NY, USA, pp 13–22
        https://link.springer.com/article/10.1007/s13042-017-0762-9
'''


from scipy import spatial
import pandas as pd
import numpy as np
from knn_cb import split_numerical_id
from sklearn import preprocessing
import math

def pid_matrix(test_data):
    '''
    Create dataframe with rows for each pid, columns for each track(pos), values for each track_uri
    
    Parameters
    ----------
    test_data : TYPE pandas dataframe

    Returns
    -------
    Dataframe
    '''
    pid_matrix = pd.pivot(test_data,values = 'track_uri',index = 'pid',columns = 'pos')
    
    return pid_matrix


def Novelty_calc(track_popularity, prediction):
    output_list = []
    sum_popularity = track_popularity.iloc[:,1].to_numpy().sum()
    for index, pred in prediction.iterrows():
        # print(index)
        novelty_list = []
        for track in pred:
            pop = track_popularity[track_popularity['track_uri']==track]['track_name']
            if pop.size == 0: 
                novelty = 0
            else:
                novelty = 1 - math.log(pop.values[0],2) /sum_popularity
            novelty_list.append(novelty)
        output_list.append(np.sum(novelty_list))
    return np.mean(output_list) , output_list

def rank_track_popularity(total_data):
    # Find the top N artist - based on popularity
    tracks = total_data[['track_name','track_uri']].groupby('track_uri').count()
    top_tracks = tracks.sort_values(['track_name'], ascending=False)
    top_tracks = top_tracks.reset_index()
    return top_tracks

prediction = pd.read_csv('output.csv', header = None)

# Read Data and drop duplicated tracks
train_data = pd.read_csv('training.csv')
train_data = train_data.reset_index(drop = True)

test_data = pd.read_csv('testing.csv')
test_seed_data = test_data[test_data['pos'] == 0]

total_data = pd.concat([train_data, test_data])

# split the input data into numerical features and identity features
data_numerical, data_id = split_numerical_id(total_data)

# scale and normalize the data
scaler = preprocessing.StandardScaler().fit(data_numerical)

train_id = data_id[:train_data.shape[0]].reset_index(drop = True)
test_id = data_id[train_data.shape[0]:].reset_index(drop = True)


pid_matrix = pid_matrix(test_data)

track_popularity = rank_track_popularity(total_data)
track_popularity = track_popularity[track_popularity['track_name']>1]

output, output_list = Novelty_calc(track_popularity, prediction)

print('Novelty:',output)
