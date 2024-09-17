'''
Data pre-processing. 
Cleaning of the data, removing missing entries, and duplicates. Visualization of the spread across classes. 
'''

import pandas as pd
import numpy as np

'''
Add the path to the raw data here. The coloumn names have not been changes w.r.t. the source data. If you do so, make sure to edit the code appropriately.
'''
path = ''

def load_data(path):
    '''
    Here we just load the data, and extract the relevant coloumns - the SMILES string and the corresponding LD50.
    '''
    data = pd.read_csv(path)
    tox = data[['Canonical SMILES', 'Toxicity Value']]
    return tox

def remove_null(data):
    '''
    Here we just remove the null entires.
    '''
    print('Initial datapoints: ' + str(len(data['Canonical SMILES'])))
    cleaned_data = data[data['Canonical SMILES'].notnull() & data['Toxicity Value'].notnull()]
    print('Final datapoints after removing null: ' + str(len(cleaned_data['Canonical SMILES'])))
    return cleaned_data

def remove_duplicates(data):
    '''
    Here we just remove the duplicate entries.
    '''
    print('Initial datapoints: ' + str(len(data['Canonical SMILES'])))
    cleaned_data = data.drop_duplicates(keep = 'first', inplace = False)
    print('Final datapoints after removing duplicates: ' + str(len(cleaned_data['Canonical SMILES'])))
    return cleaned_data

def categorize(data):
    '''
    Here we categorize the data according to the metrics specified in the manuscript.
    '''
    categories = []
    toxicity = data['Toxicity Value']
    for vals in toxicity:
        if vals < 50:
            categories.append(-1)
        elif vals >= 50 and vals < 2000:
            categories.append(0)
        elif vals >= 2000:
            categories.append(1)
    data['Category'] = categories
    
    return data

def run(path):
    data = load_data(path)
    notnulldata = remove_null(data)
    nonduplicatedata = remove_duplicates(notnulldata)
    category_data = categorize(nonduplicatedata)
    category_data.to_csv('oral_data_cleaned.csv')

    categories = category_data['Category']
    fatal = 0
    toxic = 0
    safe = 0
    for category in categories:
        if category == -1:
            fatal += 1
        elif category == 0:
            toxic += 1
        elif category == 1:
            safe += 1
    print('Number of fatal molecules: ' +  str(fatal))
    print('Number of toxic molecules: ' +  str(toxic))
    print('Number of safe molecules: ' +  str(safe))
    return 'Success'

success = run(path)
print(success)