'''
EVALUATION: INTRA-LIST SIMILARITY USING CONSINE SIMILARITY:
    
    REFERENCE: 
        https://github.com/statisticianinstilettos/recmetrics
        Ziegler, Cai-Nicolas, et al. “Improving Recommendation Lists through Topic Diversification.” Proceedings of the 14th International Conference on World Wide Web, ACM, 2005, pp. 22–32, doi:10.1145/1060745.1060754.
'''


from scipy import spatial
import pandas as pd
import numpy as np
from knn_cb import split_numerical_id
from sklearn import preprocessing

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


def ILS_calc(total_data, prediction, scaler):
    
    output_list = []
    for index, pred in prediction.iterrows():
        # print(index)
        pred_total = total_data[total_data['track_uri'].isin(pred)].drop_duplicates('track_uri').reset_index(drop = True)
        pred_numerical, pred_id = split_numerical_id(pred_total)
        pred_numerical = pd.DataFrame(scaler.transform(pred_numerical))
        similarity = 1 - spatial.distance.pdist(pred_numerical, metric='cosine')

        output_list.append(np.sum(similarity) / 2)
    return np.mean(output_list) , output_list


prediction = pd.read_csv('output.csv', header = None)

# Read Data and drop duplicated tracks
train_data = pd.read_csv('training.csv')
train_data = train_data.drop_duplicates('track_uri').reset_index(drop = True)

test_data = pd.read_csv('testing.csv')
test_seed_data = test_data[test_data['pos'] == 0]

total_data = pd.concat([train_data, test_data])

# split the input data into numerical features and identity features
data_numerical, data_id = split_numerical_id(total_data)

# scale and normalize the data
scaler = preprocessing.StandardScaler().fit(data_numerical)

pid_matrix = pid_matrix(test_data)

output, output_list = ILS_calc(total_data, prediction, scaler)

print('Intra-list Similarity:',output)
