#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import pandas as pd
import numpy as np
import pickle
import warnings
# warnings.filterwarnings('ignore') # Ignore warnings
from lifelines import CoxPHFitter, AalenAdditiveFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
import copy


# In[2]:


# Target columns
time = 'OS'
event = 'event_OS'

# Secondary Target columns
time2 = ''
event2 = ''

# File locations
clinical_loc = '/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/clinical_combined_RO.csv'
pet_loc = '/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/pet_combined_RO.csv'
response_loc = '/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/response_combined_RO.csv'
folds_loc = '../Model Training and Evaluation/Folds/hnc_combined.pkl'

# Read CSV files; using combined outliers
clinical = pd.read_csv(clinical_loc, index_col='ID')
pet = pd.read_csv(pet_loc, index_col='ID')
response = pd.read_csv(response_loc, index_col='ID')


# In[3]:


# Load file with folds
with open(folds_loc, 'rb') as handle:
    folds = pickle.load(handle)

# Separate train and test folds
train_folds = folds['train']
test_folds = folds['test']

# Define fold names
fold_names = list(train_folds.keys())

# Combine explanatory features to one df
X_df = pd.concat([clinical, pet], axis=1)


# In[4]:


response


# In[5]:


from lifelines.statistics import logrank_test

res = {}

for col in X_df.columns:
    
    # Define current df
    curr_df = pd.concat([X_df[col], response[[time, event]]], axis=1)
    
    # Encode numerical to binary
    if curr_df[col].nunique() > 2:
        med = curr_df[col].median()
        key = f'{col} > {med}'
        curr_df[key] = np.where(curr_df[col] > med, 1, 0)        
    else:
        key = col
    
    # Define event series
    E0 = curr_df[curr_df[key] == 0][event]
    E1 = curr_df[curr_df[key] == 1][event]
    
    # Define survival time series
    T0 = curr_df[curr_df[key] == 0][time]
    T1 = curr_df[curr_df[key] == 1][time]
    
    # Logrank test
    results = logrank_test(T0, T1, event_observed_A=E0, event_observed_B=E1)
    
    
    
    c_scores = []
    for fold_name in train_folds:
        
        # Find train and test indices
        train_names = train_folds[fold_name]
        test_names = test_folds[fold_name]
        
        # Define train and test splits
        train_data = curr_df.loc[train_names][[key, time, event]]
        test_data = curr_df.loc[test_names][[key, time, event]]
        
        try:
            # Train model
            model = CoxPHFitter(penalizer=0.1, l1_ratio=0)
            model.fit(train_data, duration_col=time, event_col=event)

            # Get predictions
            y_pred = model.predict_partial_hazard(test_data)
            
            # Calculate C-index
            c_scores.append(concordance_index(test_data[time], -y_pred, test_data[event]))
            
        except:
            c_scores.append(np.nan)
            
    count0 = np.sum(np.where(curr_df[key] == 0, 1, 0))   
    surv_count0 = E0.sum()
                    
    count1 = np.sum(np.where(curr_df[key] == 1, 1, 0))
    surv_count1 = E1.sum()
    res[key] = {
        'p-value': f'{results.p_value:.3f}', 
        'Null-Hypthesis': ('Reject' if results.p_value < 0.05 else 'Approve'),
        # 'test_statistic': f'{results.test_statistic:.3f}',
        'c-index': f'{np.mean(c_scores):.2f} +- {np.std(c_scores):.2f}',
        '0': count0,
        '1': count1,
        '0_failures': int(surv_count0),
        '1_failures': int(surv_count1)
    }     


# In[6]:


results_df = pd.DataFrame.from_dict(res).T.sort_values(by='c-index', ascending=False)#.head(15)

# results_df['test_statistic'] = np.where(results_df['p-value'] > 0.05, 'Reject', 'Approve')

# Assigining header names
results_df.columns = [
    ['Logrank Test','Logrank Test','Univariate CPH', 'Group Distributions', 'Group Distributions',
     'Events Per Group', 'Events Per Group'],
    ['P-Value','Null-Hypothesis',"Harrell's C-index", '0', '1', '0', '1']
]


results_df


# In[ ]:




