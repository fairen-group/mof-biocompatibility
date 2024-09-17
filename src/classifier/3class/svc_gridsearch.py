###############################################################
#
# This script is for coarse-grid hyperparameter optimisation of the support vector classifier, to focus in on a good set of hyperparameters for further optimisation.
# While both RandomizedSearch and GridSearch are compatible - you may choose an approach depending on the need.
# This could take a while depending on the parameter grid (of the order of dozens of hours). 
# I recommend running this script on a HPC.
# We will not show the model the independent test set - please remember. That is only for the production model.
# Because of the multi-metric scoring, followed by a refit - this takes much longer to train (even on a single set of hyperparameters) compared to the production model.
#
###############################################################

import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold 
from sklearn.svm import SVC

'''
Here, instead of a grid search, where every permutation of the hyperparameter space is tested, we perform a randomised search.
In this case, we fix the number of total tests to 32, and randomly select 32 sets of hyperparameters.
This is primary done to limit the computational time of training. 

'''

'''
Since this piece of code was run on a cluster, all the output was stored to an 'output' file.
If this is being run on a computer, the user is encouraged to remove bits of code tagged #1, #2 and #3.
'''
# f = open("output", "w") #1 - in case this is being run on a cluster, the user can also set path to relevant output directory, 
# sys.stdout = f #2

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
    
    model = SVC(random_state = 42, verbose = 1)

    '''
    We perform a 5-fold cross-validation, with a single repeat. 

    '''
    cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 1, random_state = 0)
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
        print(str(k) + " => " + str(v))

    '''
    We will also save the performance of the model on each set permutation of hyperparameters passed, just to extract general trends. 
    It may be necessary that we test out hyperparameters outside the ranges specified, in case the best performing model is on the bounds of the parameter space.
    '''    

    print('Saving CSV')
    cv_results = grid_search.cv_results_
    results = pd.DataFrame.from_dict(cv_results)
    results.to_csv('./gridsearch/svc_ip_grid_search_results.csv')
    print('CSV saved ... ')
    return best_score, best_params

'''
Now, let us load the data. 
Here the user needs to specify the filepath to where the data is stored. 
We recommend that the data is stored in an identical format to the training data that we have provided. 
'''

def data(path):
    data_featurized = pd.read_csv(path) 
    y = data_featurized['Category']
    X = data_featurized.drop(['Category'], axis = 1)
    
    # Since the train-test split is already done, no need to do anything here.

    return X, y

def run_model(path):
    X_train, y_train = data(path)
    gridsearch_scores, _ = supportvectorclassifier(X_train, y_train)
    return gridsearch_scores

path = './data/ip_data_features_selected_sampled.csv' # specify the datapath here. 
gridsearch_scores = run_model(path)
print('Highest achieved scores are: ' + str(gridsearch_scores))

# f.close() #3