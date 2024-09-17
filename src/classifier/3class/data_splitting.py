################################################################################################
#
# This script is used to: split the data into a (train + validation) + independent test set. 
# Note that the user will need to specify the path to their data in the run method. 
# Please note: if the CSV file has a column that indexes the data - please remove it. 
#
################################################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split(data):
    '''
    Input the complete dataframe and the fraction of the test set. - by default we will have 10% for test.
    This code-block will split the data and return to dataframe objects for train and test respectively.
    '''

    y = data['Category']
    X = data.drop(['Category', 'SMILES'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 14) # keep the random_state for reproducibility
    
    train_set = X_train.assign(Category = y_train)
    test_set = X_test.assign(Category = y_test)
    print('Test set has: ' + str(len(test_set.index)) + ' datapoints')
    print('Train set has: ' + str(len(train_set.index)) + ' datapoints')

    return train_set, test_set

path = './data/ip_data_features_selected.csv' # give here the data post feature selection.
data = pd.read_csv(path)
train, test = split(data)
train.to_csv('./data/ip_train_set.csv')
print('Train set saved.')
test.to_csv('./data/ip_test_set.csv')
print('Test set saved.')