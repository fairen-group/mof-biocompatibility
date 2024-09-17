###############################################################
#
# This script is for training the production model for toxicity prediction.
# These hyperparameters were obtained from the coarse-grid optimisation, followed by fine-grid. 
# Model training with cross-validation, evaluated across multiple metrics. Followed by evaluation against an independent test set.
# In the end - the reciever-operator characteristics are calculated in a one-vs-rest scheme.
# Minor tweaks to the hyperparameters may marginally improve metrics.
# The code is heavily commented to improve interpretability. 
#
###############################################################

import pandas as pd
import numpy as np

'''
The 'pickle' library will come in handy to save the machine learning model. 
This saves time, as we can use this model whenever we want, without needing to retrain it. 
Moreover, given additional data, we can further optimise this model.
'''

import pickle

import time
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import make_scorer, matthews_corrcoef, balanced_accuracy_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold 

'''
Here, we chose to calculate the Reciever Operator Characteristic (ROC) curve as well. For more insights into the motivations behind these plots,
refer to the main manuscript. 
'''

import matplotlib.pyplot as plt


'''
Since this piece of code was run on a cluster, all the output was stored to an 'output' file.
If this is being run on a computer, the user is encouraged to remove bits of code tagged #1, #2 and #3.
'''

# 1
# f = open("output", "w")
# 2 
# sys.stdout = f

'''
Let us define a function where the model training takes place. 
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
    
    scores = [] # list for storing the model metrics (mean)
    scores_std = [] # list for storing the model metrics (std)

    '''
    The hyperparameters inputted in the model in principle should yield the best metrics. Small tweaks may improve performance, but marginally at best.
    '''
    # keep the random_state to reproduce the exact results.
    model = RandomForestClassifier(n_jobs = -1, random_state = 42, n_estimators = 725, 
                                   max_depth = 32, min_samples_split = 2, min_samples_leaf = 1)
    #model = HistGradientBoostingClassifier(random_state = 42, learning_rate = 0.01, max_iter = 500, max_leaf_nodes = 30,
    #                                       min_samples_leaf = 30, l2_regularization = 0.0)
    
    print('Starting cross-validation...')

    '''
    We perform a 10-fold cross-validation, with a single repeat - as is recommended for large datasets.     
    '''

    cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 1, random_state = 42)
    n_scores = cross_validate(model, X, y, scoring = scoring, cv = cv)
    print('Cross-validation completed...')
    print('Saving scores...')

    '''
    Once the cross-validation is complete, we can save the scores of the model. These scores will come in handy for some future analysis.
    '''

    scores.append(np.mean(n_scores['test_accuracy']))
    scores_std.append(np.std(n_scores['test_accuracy']))
    accuracy = n_scores['test_accuracy']

    scores.append(np.mean(n_scores['test_balanced_accuracy']))
    scores_std.append(np.std(n_scores['test_balanced_accuracy']))
    balanced_accuracy = n_scores['test_balanced_accuracy']

    scores.append(np.mean(n_scores['test_f1_macros']))
    scores_std.append(np.std(n_scores['test_f1_macros']))
    f1 = n_scores['test_f1_macros']

    scores.append(np.mean(n_scores['test_MCC']))
    scores_std.append(np.std(n_scores['test_MCC']))
    MCC = n_scores['test_MCC']

    print('Scores saved...')

    '''
    Finally, we fit the model on the training data. 
    '''

    model.fit(X, y)

    return scores, scores_std, model, accuracy, balanced_accuracy, f1, MCC

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
    Since the train-test split was already done, no need to do it here. The cross-validation will take care of the validation set.
    '''
    
    return X, y

'''
Let us bring it all together and run the model now. 
'''

def run_model(path, path_test):
    X_train, y_train = data(path)
    start_train = time.time() # let us time to training.
    cross_validation_scores, scores_std, model, accuracy, balanced_accuracy, f1, MCC = randomforestclassifier(X_train, y_train)
    end_train = time.time()
    duration = end_train - start_train

    '''
    Once the model is trained on the training data, it is necessary to see if it is able to retain its accuracy on the testing data.
    Moreover, we will extract useful metrics such as the true positive and false positive rates from these results - refer to the main manuscript for more context.
    '''

    testdata = pd.read_csv(path_test) # load the independent test data.
    y_test = testdata['Category'] # Extract the category
    X_test = testdata.drop(['Category'], axis = 1)

    y_pred = model.predict(X_test) # perform predictions on the test data. 
    y_proba = model.predict_proba(X_test) # we will need this for ROC-AUC
    classes = model.classes_

    print('Cross-validation accuracy is: ' + str(cross_validation_scores[0]) + ' +/- ' + str(scores_std[0]))
    print('Cross-validation balanced accuracy is: ' + str(cross_validation_scores[1]) + ' +/- ' + str(scores_std[1]))
    print('Cross-validation F1 score is: ' + str(cross_validation_scores[2]) + ' +/- ' + str(scores_std[2]))
    print('Cross-validation MCC is: ' + str(cross_validation_scores[3]) + ' +/- ' + str(scores_std[3]))

    test_score = accuracy_score(y_test, y_pred) 
    test_score_balanced = balanced_accuracy_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_proba[:, 1]) 

    print('Accuracy on the independent test data is: ' + str(test_score) + ' (balanced accuracy : ' + str(test_score_balanced) + ', ROC-AUC : ' + str(test_roc_auc) + ')')

    return cross_validation_scores, test_score, model, X_test, y_test, y_proba, classes, accuracy, balanced_accuracy, f1, MCC, duration

'''
Based on the model results, let us calculate the true positive and false positive rates - these will help us construct the confusion matrix - and plot ROC curves.
'''

def calculate_tpr_fpr(y_true, y_pred):
    '''
    Calculates the true positive rate and false positive rate 
    Constructs a confusion matrix - refer to manuscript and supporting information for underlying theory.
    Returns true positive rate (tpr) and false positive rate (fpr).
    '''

    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0, 0] # TRUE NEGATIVES
    FP = cm[0, 1] # FALSE POSITIVES
    FN = cm[1, 0] # FALSE NEGATIVES
    TP = cm[1, 1] # TRUE POSITIVES

    tpr = TP/(TP+FN) # sensitivity (recall)
    fpr = 1 - TN/(TN+FP) # 1 - specificity

    return tpr, fpr

def get_all_roc_coordinates(y_true, y_proba):
    '''
        Calculates all the ROC curve coordinates (tpr and fpr) by considering each point as a threshold for the prediction.

        y_true: true labels
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
    
    for i in range(len(classes)): # both graphs will be the same here - as it is a binary classification. 
        c = classes[i] # -1: more toxic and 1: less toxic.
        cat = []
    
        for y in y_test: # loop for binarizing the classes.
            if y == c:
                cat.append(1)
            else:
                cat.append(0)
    
        proba = y_proba[:, i] # extract the y_proba for all datapoints in the class.

        tpr, fpr = get_all_roc_coordinates(cat, proba) 
        tpr_fpr[c] = [tpr, fpr] # we store tpr-fpr for each class.
        roc_auc_ovr[c] = roc_auc_score(cat, proba) # we store ROC-AUC for each class.
    
    return tpr_fpr, roc_auc_ovr

def plot(tpr, fpr, cat, auc):
   '''
   Plot TPR vs FPR curve. Add the line for a random chance. Also add the AUC score.
   '''

   plt.figure(figsize = (3, 3))
   ax = plt.axes()
   ax.scatter(fpr, tpr, s = 0.4, color = 'orange', linewidth = 0.4)
   ax.text(1, 0, str(round(auc, 3)), horizontalalignment = 'right', verticalalignment = 'bottom')
   #ax.fill_between(fpr, tpr, 0, color = 'yellow', alpha = 0.3)

   lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
   ax.plot(lims, lims, '--', alpha = 0.7, color = 'black', zorder = 99, linewidth = 0.8)

   ax.set_xlim(lims)
   ax.set_ylim(lims)

   ticks = [0, 0.5, 1]
   ax.set_xticks(ticks)
   ax.set_yticks(ticks)

   plt.savefig('./roc_auc/ip_roc_auc_' + str(cat) + '.png', format='png', dpi = 300)

path = './data/ip_train_set.csv' # add your filepath here.
path_test = './data/ip_test_set.csv'
cross_validation_scores, test_score, model, X_test, y_test, y_proba, classes, accuracy, balanced_accuracy, f1, MCC, duration = run_model(path, path_test)

print('Calculating ROC curve.')
tpr_fpr, roc_auc_ovr = roc_curve(classes, y_test, y_proba)

for i in range(len(classes)): # PLOT ROC_AUC curve for each class. 
    print('Class : ' + str(classes[i]))
    tpr = tpr_fpr[classes[i]][0]
    fpr = tpr_fpr[classes[i]][1]
    auc = roc_auc_ovr[classes[i]]
    plot(tpr, fpr, classes[i], auc)

filename = 'ip_finalized_model.sav'
saved_model = './models/' + filename
pickle.dump(model, open(saved_model, 'wb'))
print('Model saved.')

print('Training took %.2f seconds' % duration)
#f.close() #3