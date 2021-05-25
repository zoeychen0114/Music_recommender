import pandas as pd
from knn_cb_popularity_single import knn_run

def predict(seed_uri, liked):
    track_data = pd.read_csv('Track_Library.csv')   
    
    if liked == True: # If is second round, and the user liked a song
        previous_prediction = pd.read_csv('prediction.txt')
        
        ###### REPLACE THIS LINE OF CODE WITH W&D PREDICTION
        new_prediction = knn_run(seed_uri, num_neighbor = 20)
        
        new_prediction = new_prediction[['artist_name', 'track_name', 'track_uri', 'score']]
        
        # Merge the two recommendations together
        new_prediction = pd.concat([new_prediction[:10], previous_prediction[:10]]).drop_duplicates('track_uri').reset_index(drop = True)
        # Exclude the seed_uri itself
        new_prediction = new_prediction[new_prediction['track_uri'] != seed_uri]
        
        new_prediction = new_prediction.sort_values(by = ['score'], ascending = False)

    
    elif liked == False: # If is second round, and the user disliked a song
        previous_prediction = pd.read_csv('prediction.txt')
        # Exclude the seed_uri disliked
        new_prediction = previous_prediction[previous_prediction['track_uri'] != seed_uri]
        
        
    else: # If is first round
    
        ###### REPLACE THIS LINE OF CODE WITH W&D PREDICTION
        new_prediction = knn_run(seed_uri, num_neighbor = 20)
        
        new_prediction = new_prediction[['artist_name', 'track_name', 'track_uri', 'score']]
        
       
    # Save the first round result to combine with second round
    new_prediction.to_csv('prediction.txt', index = False)
    
    # print('is liked?',liked)
    return new_prediction[:10]


if __name__ == '__main__':
    # sample output
    seed_uri = '5ChkMS8OtdzJeqyybCc9R5'
    print('Input Seed:', seed_uri)
    output = predict(seed_uri, "")
    