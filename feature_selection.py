'''
This script is used to select the 'K' best features using an Analysis of Variance (ANOVA) test. 
Note that the user will need to specify the path to their data in the run method. 
Also, the user needs to specify the number of features that need to be selected ('K')

'''

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


def feature_selection(path, K):
    # let us load the data first
    X_featurized = pd.read_csv(path) 
    y_RFE = X_featurized['Category']
    X_RFE = X_featurized.drop(['Category'], axis = 1)

    # selection of K features
    selector = SelectKBest(f_classif, k = K) 
    selector.fit(X_RFE, y_RFE)

    col_ind = selector.get_support(indices = True) 
    X_K = X_featurized.iloc[:, col_ind]
    return X_K

def run(path, K): 
    X_updated = feature_selection(path = path, K = K)
    X_updated.to_csv('K_features.csv') # the user may choose a custom directory, else will be saved in the parent directory.
    return X_updated
'''
uncomment the lines below if you want to directly run the code from here; if you are calling this function in a separate script, keep it commented.
Remember to specify both the filepath to the data, and the value of K.
'''
# path = '' # enter your filepath here
# K = 100 # enter the desired number of features.

# print('Beginning run')
# X_updated = run(path = path, K = K)
# print('Run completed')
# print('File saved')
