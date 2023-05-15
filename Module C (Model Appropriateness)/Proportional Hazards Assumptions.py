#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Imports
import pandas as pd
import numpy as np
import pickle
import warnings
# warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter


# In[7]:


# OxyTarget
X_df = pd.read_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/oxytarget/oxytarget_RO.csv', index_col='ID')
response = pd.read_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/oxytarget/oxytarget_RO_response.csv', index_col='ID')

# Target columns
time = 'Time until OS event'
event = 'OS event'

# Folds loc
folds_loc = 'Folds/oxytarget.pkl'

# Subset loc
subset_loc = '../Exploratory Data Analysis/Feature Subsets/OxyTargetFeatures.pkl'

if subset_loc:
    with open(subset_loc, "rb") as openfile:
        subset = pickle.load(openfile)
else:
    subset = None
    
if subset:
    X_df = X_df[subset]

surv_df = pd.concat([X_df, response[[time, event]]], axis=1)
cph = CoxPHFitter(penalizer=0.1, l1_ratio=0)
cph.fit(surv_df, duration_col=time, event_col=event)
cph.check_assumptions(surv_df, show_plots=True)


# In[4]:


# HNC
clinical = pd.read_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/clinical_combined_RO.csv', index_col='ID')
pet = pd.read_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/pet_combined_RO.csv', index_col='ID')
response = pd.read_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/response_combined_RO.csv', index_col='ID')

# Target columns
time = 'OS'
event = 'event_OS'

surv_df = pd.concat([clinical, pet, response[[time, event]]], axis=1)
cph = CoxPHFitter(penalizer=0.1, l1_ratio=0)
cph.fit(surv_df, duration_col=time, event_col=event)
cph.check_assumptions(surv_df, show_plots=True)


# In[ ]:




