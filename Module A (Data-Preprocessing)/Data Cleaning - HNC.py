#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


# Set path
import os
import sys

# Change this to the library for the utils folder on your computer
sys.path.insert(0, os.path.abspath('/Users/mikkorekstad/Skole/Master/repository/Data Preparation/utils'))

# Basic python tools
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Formatting tools
import formatting_tools

# Sklearn preprocessing tools
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


# # Loading Data

# In[2]:


# Read data from original file
response = pd.read_excel('/Users/mikkorekstad/Skole/master_data/raw_data/head_neck/ous/response_ous.xlsx', index_col='patient_id')
n_rows = response.shape[0]
n_cols = response.shape[1]
print(f'Read df in with {n_rows} rows and {n_cols} columns.')

response.head()


# In[3]:


# Find missing values for response DF
response.isnull().sum()


# In[4]:


response['event_OS'].sum()

response.sum()
# In[5]:


response.sum()


# In[6]:


# Read data from original file
clinical = pd.read_excel('/Users/mikkorekstad/Skole/master_data/raw_data/head_neck/ous/clinical_ous.xlsx', index_col='patient_id')
n_rows = clinical.shape[0]
n_cols = clinical.shape[1]
print(f'Read df in with {n_rows} rows and {n_cols} columns.')

clinical.head()


# In[7]:


# Find missing values for clinical DF
clinical.isnull().sum()


# In[8]:


# Read data from original file
pet = pd.read_excel('/Users/mikkorekstad/Skole/master_data/raw_data/head_neck/ous/pet_parameters_ous.xlsx', index_col='patient_id')
n_rows = pet.shape[0]
n_cols = pet.shape[1]
print(f'Read df in with {n_rows} rows and {n_cols} columns.')

pet.head()


# In[ ]:





# In[9]:


pet.isnull().sum()


# - Target columns looks OK.
# - Only two binary columns with missing data.
# - All other columns appears to be of a interpretable format.

# # Handling Missing Values
# - For columm hpv_related: NaN suggests unknown HPV status
# - For columm uicc8_III-IV: This value is also unknown for patients with unknown HPV status.
# 
# We are considering the following options for handling these missing values:
# - Create a new column for hpva_status_unknown
# - Imputing the missing values for one of the columns, and carry that information to the other.
# - Remove both columns, because 52 / 197 patients makes up a large proportion.
# 
# We will start by trying the first option, because we want to keep the information from the two columns!

# In[10]:


clinical['hpva_status_unknown'] = np.where(clinical['hpv_related'].isnull(), 1, 0)
clinical.fillna(0, inplace=True)


# In[11]:


clinical


# # Numerical Columns

# In[12]:


clinical.nunique()


# In[13]:


pet.nunique()


# In[14]:


pd.concat([clinical[['age', 'pack_years']], pet, response[['OS', 'DFS', 'LRC']]], axis=1).describe()


# - Surprisingly high value for max value in column pack_years. 
#     - We will look further into this in the data exploration.
#     
# - MTV and TLG have seemingly high variance. SUVpeak smaller variance in comparison.    
#  
# - endpoint loco-regional control and disease-free survival has somewhat similar statistics, and OS is quite a bit higher.
# 
# All these numerical features will be interesting to further explore in the data exploration notebook.

# # Change Index Names

# In[15]:


response_names = [f'HeadNeck{i:03d}' for i in response.index]
pet_names = [f'HeadNeck{i:03d}' for i in pet.index]
clinical_names = [f'HeadNeck{i:03d}' for i in clinical.index]

num_equal = 0
for r, p, c in zip(response_names, pet_names, clinical_names):
    if r == p and p == c:
        num_equal += 1
    else:
        print('Not Equal')
        

if len(response_names) == len(pet_names) and len(pet_names) == len(clinical_names):
    print('Equal length OK.')
print(f'{num_equal} equal index names, exptected {len(response_names)}')

response.index = pd.Index(response_names, name='ID')
pet.index = pd.Index(response_names, name='ID')
clinical.index = pd.Index(response_names, name='ID')


# # Now save the cleaned data set

# In[16]:


response.to_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/response.csv')
clinical.to_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/clinical.csv')
pet.to_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/pet.csv')



