'''
Unique features
'''
import pandas as pd
import numpy as np

intraperitoneal = pd.read_csv('/Users/dhruvmenon/Documents/PhD/003_ml_biocompatibility/source_code/data_sampled_random_ADASYN_110_3class.csv')
oral = pd.read_csv('/Users/dhruvmenon/Documents/PhD/003_ml_biocompatibility/source_code/github/oral_data_sampled_random_ADASYN_3class_110.csv')

ip_columns = []
oral_columns = []

for column, _ in intraperitoneal.items():
    ip_columns.append(column)

for column, _ in oral.items():
    oral_columns.append(column)

for column in ip_columns:
    if column not in oral_columns:
        print('Feature ' + column + ' is unique to ip')

for column in oral_columns:
    if column not in ip_columns:
        print('Feature ' + column + ' is unique to oral')