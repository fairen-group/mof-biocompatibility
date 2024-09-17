###############################################################
#
# In this script, we will sample datapoints so as to reduce class imbalance - to avoid any bias during machine learning model training.
# As discussed in the manuscript, the majority class will be undersampled, while the minority class will be oversampled. 
# While, this code has been written to implement some different sampling algorithms, random undersampling and ADASYN for oversampling, seemed to work well.
# Remember to: !pip install imblearn
# 
# I find data sampling to be a bit of an art. You want the classes to be as balanced as possible, but you don't want to undersample or oversample to much.
# I tried several different sampling strategies - however, it is entirely possible that other strategies may work better (but likely marginally). 
#
###############################################################

from imblearn.over_sampling import SMOTE, ADASYN 
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pandas as pd

'''
Needless to say, there are several algorithms both for undersampling and oversampling (refer to imblearn documentation), 
here, most relevant algorithms have been tested.

First, we define a function for random undersampling.
'''

def undersample_random(X, y):
    '''
    The sampling strategies are hard-coded for our dataset. It will need to be changed based on the use-case.
    '''
    # defining the strategy to undersample the data.
    strategy = {-1: 4000, 0: 4000, 1: 4000} # define this based on the final size of each class.
    undersample = RandomUnderSampler(sampling_strategy = strategy, random_state = 42) # random state for reproducibility.
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
    oversample = ADASYN(n_neighbors = 10, random_state = 42) # may be optimised, not super critical in the present context.
    X_new, y_new = oversample.fit_resample(X, y)

    # summarize distributions
    counter = Counter(y_new)
    for k, v in counter.items():
        per = v/len(y) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    return X_new, y_new

def run(path):
    '''
    Load the dataset. 
    Extract X and y (remove SMILES).
    Pass to the sampler functions.
    Return the sampled datasets.
    '''
    dataset = pd.read_csv(path)

    '''
    First, we remove all toxic molecules out of the dataset. 
    We then, sample required datapoints from these toxic molecules. 
    Add these to the original dataset of safe and fatal molecules.
    We then run ADASYN.
    Finally, we run the undersampler to equalise class labels.
    '''

    data_toxic = dataset[dataset['Category'] == 0] # extract toxic data.
    dataset_small = dataset.drop(data_toxic.index) # remove toxic data from larger dataset.

    # here is where the 'art' comes in. You want your sample points to be not too less nor too many. 
    data_toxic_sampled = data_toxic.sample(n = 6000, random_state = 42) # sample required datapoints. 

    dataset_updated = pd.concat([dataset_small, data_toxic_sampled], ignore_index = True)
    dataset_updated = dataset_updated.sample(frac = 1, random_state = 42).reset_index(drop = True) # shuffle the dataset.
    
    y = dataset_updated['Category']
    X = dataset_updated.drop(['Category'], axis = 1)

    # X_under, y_under = undersample_random(X, y)
    X_over, y_over = oversample_ADASYN(X, y)
    
    '''
    ADASYN produces slightly unequal classes. If you want to explicitly have equal classes, then run the undersampler to equalize.
    Else, proceed as is. Here we run the undersampler. 
    '''
    
    # Make sure to check the strategy use in the function. 
    X_under, y_under = undersample_random(X_over, y_over) # sample according to the strategy

    dataset_sampled = X_under.assign(Category = y_under)
    dataset_sampled.to_csv('./data/oral_data_features_selected_sampled_3.csv')
    return dataset_sampled

path = './data/oral_train_set.csv'
_ = run(path)