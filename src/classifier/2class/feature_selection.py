################################################################################################
#
# This script is used to select the 'K' best features using an Analysis of Variance (ANOVA) test. 
# Note that the user will need to specify the path to their data in the run method. 
# Also, the user needs to specify the number of features that need to be selected ('K')
# Please note: if the CSV file has a column that indexes the data - please remove it. 
#
################################################################################################

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif


def feature_selection(path, K = 110): # default K = 110 (best results during benchmarking).
    '''
    IMPORTANT: If the CSV files contain an INDEX column - remove it manually before running this script.
    '''
    # let us load the data first
    X_featurized = pd.read_csv(path) 
    y_RFE = X_featurized['Category']
    smiles = X_featurized['SMILES']
    X_RFE = X_featurized.drop(['Category', 'SMILES'], axis = 1) # remove these two columns before feature selection.

    # selection of K features
    selector = SelectKBest(f_classif, k = K) 
    selector.fit(X_RFE, y_RFE)
    print('Feature selection complete.')

    col_ind = selector.get_support(indices = True) # extract the column names for the KBest features.
    X_K = X_featurized.iloc[:, col_ind] # extract the KBest features.
    
    # don't forget to add back the target and SMILES columns.
    X_K_SMILES = X_K.assign(SMILES = smiles)
    X_K_final = X_K_SMILES.assign(Category = y_RFE)

    return X_K_final



def run(path, K = 110): 
    X_updated = feature_selection(path = path, K = K)
    X_updated.to_csv('./data/oral_features_selected.csv') # the user may choose a custom directory, else will be saved in the parent directory.
    return X_updated


'''
uncomment the lines below if you want to directly run the code from here; if you are calling this function in a separate script, keep it commented.
Remember to specify both the filepath to the data, and the value of K.
See my comment above about the index column.
'''

path = './data/oral_featurised_toxic.csv' # enter your filepath here
K = 110 # enter the desired number of features.

print('Beginning run')
X_updated = run(path = path, K = K)
print('Run completed')
print('File saved')
