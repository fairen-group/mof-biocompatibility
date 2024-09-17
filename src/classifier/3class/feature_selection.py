###############################################################
#
# This script is for selecting features most-correlated to the target (i.e. toxicity)
# We use the standard SelectKBest approach based on the Analysis of Variance (ANOVA) test.
# While we got the best performance at K = 110, you are free to experiment for youself. 
# Just change the value of 'K' accordingly.
#
###############################################################

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

def feature_selection(path, K):
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

def run(path, K): 
    X_updated = feature_selection(path = path, K = K)
    X_updated.to_csv('./data/oral_data_features_selected.csv') # change name accordingly.
    return X_updated


path = './data/oral_data_featurised.csv' # enter your filepath here
K = 110 # enter the desired number of features.

print('Beginning run')
X_updated = run(path = path, K = K)
print('Run completed')
print('File saved')
