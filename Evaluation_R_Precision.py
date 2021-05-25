'''
EVALUATION: R-Precision:
    
    REFERENCE: https://github.com/phoebewong/spotify-teamNPK/blob/master/src/baseline_model_recap.ipynb
'''

import pandas as pd
import numpy as np
# import knn_cb


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

def r_precision(prediction, val_set):
    '''
    Parameters
    ----------
    prediction : 
        dataframe - 1st col:seed traks; col 2~11: pred tracks.
    val_set : 
        dataframe - ground truth. the rest of track in the same playlist as the seed.

    Returns
    -------
    score : 
        float - the proportion of predicted tracks that exist in the true playlist.
        '''
    def r_precision_formula(prediction, val_set):
        score = np.sum(val_set.isin(prediction))/val_set.shape[0]
        return score

    # Calculation for r-precision for each playlist (each row)
    score = []
    for index, pred in prediction.iterrows():
        # pred = val_set.iloc[index,:10] # A simple test to overwrite pred with ground truth
        r_prec = r_precision_formula(pred, val_set.iloc[index,:].dropna())
        score.append(r_prec)
        
    # Take mean of the r-precision across all the playlists
    return np.mean(score)

# Read the ground truth playlists
#test_uri = pd.DataFrame(result['seed'].drop_duplicates()).reset_index().drop(columns=['index'])
#test_data = pd.read_csv('data_sample.csv')
#test_data = test_data.loc[test_data['track_uri'].isin(test_uri['seed'].values.tolist())][['track_uri','pid','pos']]

#seed_df = pd.read_csv('seed_pid_tracks.csv').iloc[:,[0,2,3]]
#pid_matrix = pid_matrix(seed_df)
#seed_uri = pid_matrix.iloc[:,0]

# Read prediction results
prediction = pd.read_csv('result_0430.csv')

#prediction = pd.read_csv('result.csv', header = None).iloc[:,1:] # exclude the first track - seed

val_set = pd.read_csv('test_data_0430.csv') # exclude the first track - seed
#output_mean = r_precision(prediction, val_set)
#print('Mean R-Precision:',output_mean)

