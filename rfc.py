'''
Here, we train the best performing random forest classifier based on both the coarse grid and fine grid hyperparameter optimization for predicting the
toxicity of the organic molecule.

Minor tweaks to the hyperparameters may marginally improve metrics. 
'''

import pandas as pd
import numpy as np

'''
The 'pickle' library will come in handy to save the machine learning model. 
This saves time, as we can use this model whenever we want, without needing to retrain it. 
Moreover, given additional data, we can even optimise this model.
'''

import pickle

import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, matthews_corrcoef,  balanced_accuracy_score, accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold # RepeatedStratifiedKFold 

'''
Here, we chose to calculate the Reciever Operator Characteristic (ROC) curve as well. For more insights into the motivations behind these plots,
refer to the main manuscript. 
'''
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix 

'''
Since this piece of code was run on a cluster, all the output was stored to an 'output' file.
If this is being run on a computer, the user is encouraged to remove bits of code tagged #1, #2 and #3.
'''
f = open("output", "w") #1 - in case this is being run on a cluster, the user can also set path to relevant output directory, 
sys.stdout = f #2

'''
Let us now define a function where the model training takes place. 

'''
def randomforestclassifier(X, y):
    '''
    Below, we define the scoring functions. For more context as to why these metrics are chosen, or how they are calculated, 
    please refer to the manuscript - where these concepts are discussed in some detail. 

    '''
    scoring = {'accuracy' : make_scorer(accuracy_score),
               'balanced_accuracy' : make_scorer(balanced_accuracy_score),
               'f1_macros' : 'f1_macro',
               'MCC' : make_scorer(matthews_corrcoef)}
    scores = []

    '''
    The hyperparameters inputted in the model in principle should yield the best metrics. Small tweaks may improve performance, but marginally at best.
    '''

    model = RandomForestClassifier(n_jobs = -1, random_state = 0, n_estimators = 725, 
                                   max_depth = 32, min_samples_split = 2, min_samples_leaf = 1)
    
    print('Starting cross-validation...')

    '''
    We perform a 5-fold cross-validation, with a single repeat - as is recommended for large datasets. 

    '''

    cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 1, random_state = 0)
    n_scores = cross_validate(model, X, y, scoring = scoring, cv = cv)
    print('Cross-validation completed...')
    print('Saving scores...')

    '''
    Once the cross-validation is complete, we can save the scores of the model. These scores will come in handy for some future analysis.
    '''

    scores.append(np.mean(n_scores['test_accuracy']))
    accuracy = n_scores['test_accuracy']
    scores.append(np.mean(n_scores['test_balanced_accuracy']))
    balanced_accuracy = n_scores['test_balanced_accuracy']
    scores.append(np.mean(n_scores['test_f1_macros']))
    f1 = n_scores['test_f1_macros']
    scores.append(np.mean(n_scores['test_MCC']))
    MCC = n_scores['test_MCC']
    print('Scores saved...')
    '''
    Finally, we fit the model on the training data. 
    '''
    model.fit(X, y)
    return scores, model, accuracy, balanced_accuracy, f1, MCC

'''
Now, let us load the data. 
Here the user needs to specify the filepath to where the data is stored. 
We recommend that the data is stored in an identical format to the training data that we have provided. 

'''

def data(path):
    data_featurized = pd.read_csv(path) # modify accordingly
    y = data_featurized['Category']
    X = data_featurized.drop(['Category'], axis = 1)
    '''
    We got best results at a train-test-split of 80%/20%. 
    The user however, is free to experiment with the split. 

    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 14)

    return X_train, X_test, y_train, y_test

'''
Let us bring it all together and run the model now. 
'''
def run_model(path):
    X_train, X_test, y_train, y_test = data(path)
    cross_validation_scores, model, accuracy, balanced_accuracy, f1, MCC = randomforestclassifier(X_train, y_train)
    '''
    Once the model is trained on the training data, it is necessary to see if it is able to retain its accuracy on the testing data.
    Moreover, we will extract useful metrics such as the true positive and false positive rates from these results - refer to the main manuscript for more context.
    '''
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = model.classes_

    test_score = accuracy_score(y_pred, y_test)
    print('Cross-validation scores are: ' + str(cross_validation_scores))
    print('Score on the test data is: ' + str(test_score))

    return cross_validation_scores, test_score, model, X_test, y_test, y_proba, classes, accuracy, balanced_accuracy, f1, MCC

'''
Based on the model results, let us calculate the true positive and false positive rates - these will help us construct the confusion matrix - and plot ROC curves.
'''

def calculate_tpr_fpr(y_true, y_pred):
    '''
    Calculates the true positive rate and false positive rate 
    Constructs a confusion matrix - refer to manuscript and supporting information for underlying theory.
    '''
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0, 0] # TRUE NEGATIVES
    FP = cm[0, 1] # FALSE POSITIVES
    FN = cm[1, 0] # FALSE NEGATIVES
    TP = cm[1, 1] # TRUE POSITIVES

    tpr = TP/(TP+FN) # sensitivity
    fpr = 1 - TN/(TN+FP) # 1-specificity

    return tpr, fpr

def get_all_roc_coordinates(y_true, y_proba):
    '''
        Calculates all the ROC curve coordinates (tpr and fpr) by considering each point as a threshold for the prediction
        y_real: true labels
        y_proba: array with probabilities of each class

        returns:
        tpr_list - list of tprs representing each threshold.
        fpr_list - list of fprs representing each threshold.
    '''

    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_true, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list


def roc_curve(classes, y_test, y_proba):
    roc_auc_ovr = {}
    tpr_fpr = {}
    for i in range(len(classes)):
        c = classes[i]
        cat = []
        for y in y_test:
            if y == c:
                cat.append(1)
            else:
                cat.append(0)
        proba = y_proba[:, i]


        tpr, fpr = get_all_roc_coordinates(cat, proba)
        tpr_fpr[c] = [tpr, fpr]
        roc_auc_ovr[c] = roc_auc_score(cat, proba)
    return tpr_fpr, roc_auc_ovr

def plot(tpr, fpr):
   
    plt.figure(figsize = (3, 3))
    ax = plt.axes()
    ax.scatter(x = fpr, y = tpr)
    plt.show()

path = '' # add your filepath here.
cross_validation_scores, test_score, model, X_test, y_test, y_proba, classes, accuracy, balanced_accuracy, f1, MCC = run_model(path)
tpr_fpr, roc_auc_ovr = roc_curve(classes, y_test, y_proba)
print(tpr_fpr)
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

f.close() #3