'''
Here, we train a support vector classifer on a coarse grid of hyperparameters to optimise the chosen metrics, 
for predicting the class of toxicity of the organic ligand.

While we have kept a fairly broad range of hyperparameters - upon recommendations from the literature, 
the users can manually alter these as well. 
'''

import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, matthews_corrcoef, balanced_accuracy_score, accuracy_score
'''
Here, instead of a grid search, where every permutation of the hyperparameter space is tested, we perform a randomised search.
In this case, we fix the number of total tests to 32, and randomly select 32 sets of hyperparameters.
This is primary done to limit the computational time of training. 

'''
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

'''
Since this piece of code was run on a cluster, all the output was stored to an 'output' file.
If this is being run on a computer, the user is encouraged to remove bits of code tagged #1, #2 and #3.
'''
f = open("output", "w") #1 - in case this is being run on a cluster, the user can also set path to relevant output directory, 
sys.stdout = f #2

'''
Let us now define a function where the model training takes place. 

'''

def supportvectorclassifier(X, y):
    '''
    Below, we define the scoring functions. For more context as to why these metrics are chosen, or how they are calculated, 
    please refer to the manuscript - where these concepts are discussed in some detail. 

    '''
    scoring = {'accuracy' : make_scorer(accuracy_score),
               'balanced_accuracy' : make_scorer(balanced_accuracy_score),
               'f1_macros' : 'f1_macro',
               'MCC' : make_scorer(matthews_corrcoef)}
    
    '''
    Let us define a parameter grid that will be fed to the classifier using GridSearchCV
    User can tweak these, to test effects on model performance.

    '''
    
    param_grid = {'C' : [0.1, 1, 5, 10], 
                  'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
                  'decision_function_shape' : ['ovo', 'ovr']}
    model = SVC(n_jobs = -1, random_state = 0, verbose = 1)

    '''
    We perform a 5-fold cross-validation, with a single repeat. 

    '''
    cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 1, random_state = 0)
    cv.get_n_splits(X, y)
    print('Starting randomized search...')
    grid_search = RandomizedSearchCV(estimator = model, param_distributions = param_grid, n_iter = 32, scoring = scoring, cv = cv, refit = 'f1_macros')
    grid_search.fit(X, y)
    print('Finished randomized search...')

    '''
    Now that the randomized search is complete, we can extract the best performing model parameters and associated metrics.

    '''

    best_score = grid_search.best_score_ # returns the score from the best performing model
    print('Highest achieved scores is: f1 = ' + str(best_score))

    print('Best parameters are: ')
    best_params = grid_search.best_params_
    for k, v in best_params.items():
        print(k + " => " + str(v))

    '''

    We will also save the performance of the model on each set permutation of hyperparameters passed, just to extract general trends. 
    It may be necessary that we test out hyperparameters outside the ranges specified, in case the best performing model is on the bounds of the parameter space.

    '''    

    print('Saving CSV')
    cv_results = grid_search.cv_results_
    results = pd.DataFrame.from_dict(cv_results, orients = 'columns')
    results.to_csv('grid_search_results.csv')
    print('CSV saved')
    return best_score, best_params

'''
Now, let us load the data. 
Here the user needs to specify the filepath to where the data is stored. 
We recommend that the data is stored in an identical format to the training data that we have provided. 

'''

def data(path):
    data_featurized = pd.read_csv(path) # modify
    y = data_featurized['Category']
    X = data_featurized.drop(['Category'], axis = 1)
    '''
    We got best results at a train-test-split of 80%/20%. 
    The user however, is free to experiment with the split. 

    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)
    return X_train, X_test, y_train, y_test

def run_model(path):
    X_train, _, y_train, _ = data(path)
    gridsearch_scores, _ = supportvectorclassifier(X_train, y_train)
    return gridsearch_scores

path = '' # specify the datapath here. 
gridsearch_scores = run_model(path)
print('Highest achieved scores are: ' + str(gridsearch_scores))

f.close() #3