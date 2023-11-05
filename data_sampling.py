'''
In this script, we will sample datapoints so as to make each class equal - to avoid any bias during machine learning model training.
As discussed in the manuscript, the majority class will be undersampled, while the minority class will be oversampled. 
While, this code has been written to implement some different sampling algorithms, random undersampling and ADASYN for oversampling, seemed to work well.
Remember to: !pip install imblearn

'''
from imblearn.over_sampling import SMOTE, ADASYN # different oversampling methods tested.
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

'''
Needless to say, there are several algorithms both for undersampling and oversampling (refer to imblearn documentation), 
here, most relevant algorithms have been tested.

First, we define a function for random undersampling.
'''

def undersample_random(X, y):
    # defining the strategy to undersample the data
    strategy = {-1: 0, 0: 5000, 1: 0} # define this based on the final size of each class, here we want to undersample majority class to 5000 datapoints.
    undersample = RandomUnderSampler(sampling_strategy = strategy)
    X_new, y_new = undersample.fit_resample(X, y)

    # summarize distributions
    counter = Counter(y_new)
    for k, v in counter.items():
        per = v/len(y) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    return X_new, y_new

'''
Next, let us define functions for oversampling - first using SMOTE, second using ADASYN.

'''

def oversample_SMOTE(X, y):
    oversample = SMOTE()
    X_new, y_new = oversample.fit_resample(X, y)

    # summarize distributions
    counter = Counter(y_new)
    for k, v in counter.items():
        per = v/len(y) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    return X_new, y_new

def oversample_ADASYN(X, y):
    oversample = ADASYN(n_neighbors = 10) # may be optimised, not super critical in the present context.
    X_new, y_new = oversample.fit_resample(X, y)

    # summarize distributions
    counter = Counter(y_new)
    for k, v in counter.items():
        per = v/len(y) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    return X_new, y_new