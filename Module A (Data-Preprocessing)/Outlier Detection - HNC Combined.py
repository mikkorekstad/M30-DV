#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats


# # Load Data

# In[2]:


# Read data from original file
clinical = pd.read_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/clinical.csv', index_col='ID')
pet = pd.read_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/pet.csv', index_col='ID')
response = pd.read_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/response.csv', index_col='ID')

df = pd.concat([clinical, pet], axis=1)

# Target columns
time = 'OS'
event = 'event_OS'

# Quick overview of the data
pd.concat([df, response], axis=1).head()


# # Find Outliers based on Z - Scores
# - With the Z - score method, we are counting how many standard deviations from the mean the observation is.
# - Because the population is larger than 30, we prefare Z rather than t.
# - A Z-score higher than 3 is considered very unlikely in most scenarios, and is therefore the limit for this test.
# - In this method, we also try out separating by gender.

# In[3]:


# For the Z - score outliers, we are looking at continuous values only
binary = []
continuous = []
cols = list(df.columns)
for i, n in enumerate(df.nunique()):
    col = cols[i]
    if n > 2:
        continuous.append(col)
    else:
        binary.append(col)


# In[4]:


print(f'There are {len(continuous)} continuous variables in this data set.')
df[continuous].head()


# In[5]:


# Also print out some basic statistics
df[continuous].describe()


# In[6]:


df_male = df[df['female'] == 0]
df_male[continuous].describe()


# In[7]:


df_female = df[df['female'] == 1]
df_female[continuous].describe()


# In[8]:


thresh = 3
z_score_outliers_female = {}
outlier_counts_female = {}

z_score_outliers_male = {}
outlier_counts_male = {}

z_score_outliers_general = {}
outlier_counts_general = {}

for col in continuous:
    
    # General outliers
    z_score_outliers_general[col] = list(df[(np.abs(stats.zscore(df[col])) > thresh)].index)
    outlier_counts_general[col] = len(z_score_outliers_general[col])
    
    # Female outliers
    z_score_outliers_female[col] = list(df_female[(np.abs(stats.zscore(df_female[col])) > thresh)].index)
    outlier_counts_female[col] = len(z_score_outliers_female[col])
    
    # Male outliers
    z_score_outliers_male[col] = list(df_male[(np.abs(stats.zscore(df_male[col])) > thresh)].index)
    outlier_counts_male[col] = len(z_score_outliers_male[col])
    
#z_score_outliers = list(set(np.ravel([list(lst) for col, lst in z_score_outliers_dict.items()])))


# In[9]:


z_score_outliers_general


# In[10]:


z_score_outliers_female


# In[11]:


z_score_outliers_male


# In[12]:


# Find all conditional outliers
conditional_z_score_outliers = []
for col, lst in z_score_outliers_female.items():
    conditional_z_score_outliers.extend(list(lst))
for col, lst in z_score_outliers_male.items():
    conditional_z_score_outliers.extend(list(lst))
conditional_z_score_outliers = sorted(list(set(conditional_z_score_outliers)))

# Find all general outliers
general_z_score_outliers = []
for col, lst in z_score_outliers_general.items():
    general_z_score_outliers.extend(list(lst))    
general_z_score_outliers = sorted(list(set(general_z_score_outliers)))


# In[13]:


print('+', '=' * 15, '+', '=' * 15, '+', '=' * 11, '+')
print(f'I     {"Outlier"}     I   {"Conditional"}   I   {"General"}   I')
print('+', '=' * 15, '+', '=' * 15, '+', '=' * 11, '+')
for outlier in sorted(set(conditional_z_score_outliers + general_z_score_outliers)):
    conditional = 'X' if outlier in conditional_z_score_outliers else ' '
    general = 'X' if outlier in general_z_score_outliers else ' '
    print(f"I{outlier:^17}I{conditional: ^17}I{general: ^13}I")
print('+', '=' * 15, '+', '=' * 15, '+', '=' * 11, '+')
# conditional_z_score_outliers


# In[14]:


# Make these into lists for df
z_conditional_lst = [bool(i in sorted(conditional_z_score_outliers)) for i in list(df.index)]
z_general_lst = [bool(i in sorted(general_z_score_outliers)) for i in list(df.index)]

# Define the df
outlier_df = pd.DataFrame(z_general_lst, index=sorted(df.index), columns=['Z Score'])
outlier_df['Z Score (Conditional)'] = z_conditional_lst

outlier_df[(outlier_df['Z Score (Conditional)'] == True) | (outlier_df['Z Score'] == True)]


# In[15]:


lst = list(outlier_df[(outlier_df['Z Score (Conditional)'] == True) | (outlier_df['Z Score'] == True)].index)
df.loc[lst]


# # Quantile Outliers
# - Now we do the same thing but only with quantiles

# In[16]:


# Code inspired by https://towardsdatascience.com/outlier-detection-part1-821d714524c

def find_cuantile_outliers(df, cols, diff=1.5):
    outliers_dict = {}
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = list(df[(df[col]<(Q1-diff*IQR)) | (df[col]>(Q3+diff*IQR))].index)
        outliers_dict[col] = outliers
        
    return outliers_dict


# In[17]:


quantile_male_outliers = find_cuantile_outliers(df_male, continuous)
quantile_male_outliers


# In[18]:


quantile_female_outliers = find_cuantile_outliers(df_female, continuous)
quantile_female_outliers


# In[19]:


quantile_general_outliers = find_cuantile_outliers(df, continuous)
quantile_general_outliers 


# In[20]:


# Find all conditional outliers
conditional_quantile_outliers = []

# Add female outliers
for col, lst in quantile_female_outliers.items():
    conditional_quantile_outliers.extend(list(lst))
    
# Add male outliers
for col, lst in quantile_male_outliers.items():
    conditional_quantile_outliers.extend(list(lst))
conditional_quantile_outliers = sorted(list(set(conditional_z_score_outliers)))

# Find all general outliers
general_quantile_outliers_lst = []
for col, lst in quantile_general_outliers.items():
    general_quantile_outliers_lst.extend(list(lst))    
general_quantile_outliers_lst = sorted(list(set(general_quantile_outliers_lst)))


# In[21]:


print('+', '=' * 15, '+', '=' * 15, '+', '=' * 11, '+')
print(f'I     {"Outlier"}     I   {"Conditional"}   I   {"General"}   I')
print('+', '=' * 15, '+', '=' * 15, '+', '=' * 11, '+')
for outlier in sorted(set(conditional_quantile_outliers + general_quantile_outliers_lst)):
    conditional = 'X' if outlier in conditional_quantile_outliers else ' '
    general = 'X' if outlier in general_quantile_outliers_lst else ' '
    print(f"I{outlier:^17}I{conditional: ^17}I{general: ^13}I")
print('+', '=' * 15, '+', '=' * 15, '+', '=' * 11, '+')


# In[22]:


# Make these into lists for df
q_conditional_lst = [bool(i in sorted(conditional_quantile_outliers)) for i in list(df.index)]
q_general_lst = [bool(i in sorted(general_quantile_outliers_lst)) for i in list(df.index)]

# Define the df
outlier_df['Quantile'] = q_general_lst
outlier_df['Quantile (Conditional)'] = q_conditional_lst

# Print out the new outliers
outlier_df[(outlier_df['Quantile (Conditional)'] == True) | (outlier_df['Quantile'] == True)]


# In[23]:


lst = list(outlier_df[(outlier_df['Quantile (Conditional)'] == True) | (outlier_df['Quantile'] == True)].index)
df.loc[lst]


# # Local Outlier Factor

# In[24]:


# Information from: https://medium.com/@pramodch/understanding-lof-local-outlier-factor-for-implementation-1f6d4ff13ab9
# Code from: https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

X = df.values
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# k should be larger than min clustersize and smaller than max objects around that can be local outliers
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
#clf.fit(X)
#pred = clf.fit_predict(X)
pred = np.where(clf.fit_predict(X) == -1, True, False)
# Get NOF scores
#negative_outlier_factors = clf.negative_outlier_factor_


# In[25]:


outlier_df['LOF'] = pred
outlier_df.head()


# In[26]:


outlier_df.sum()


# # Isolation Forest
# - Good on data with high dimensionality

# In[27]:


# Code from https://scikit-learn.org/stable/modules/outlier_detection.html

from sklearn.ensemble import IsolationForest

# Fit model
clf = IsolationForest(n_estimators=10, warm_start=True, random_state=1)
clf.fit(X)  # fit 10 trees  
clf.set_params(n_estimators=20)  # add 10 more trees  
clf.fit(X)  # fit the added trees  

# 1 for an inlier and -1 for an outlier according to the LOF score and the contamination parameter.
pred = np.where(clf.predict(X) == -1, True, False)


# In[28]:


outlier_df['Isolation Forest'] = pred
outlier_df.head()


# # Conclusion
# It looks like the Isolation forest had quite a lot of outliers in this prediction.
# - Removing that many features would likely damage the performance and reliability for our models.
# - For HNC we decide to remove Z-score outliers based on the condition of gender.

# In[29]:


outlier_df.sum()


# In[30]:


outlier_df[(outlier_df['LOF'] == True)]


# In[31]:


# Create a brief summary
outlier_df['Summary'] = outlier_df['Quantile (Conditional)'] * 1 + outlier_df['Z Score (Conditional)'] * 1 + outlier_df['LOF'] * 1 + outlier_df['Isolation Forest'] * 1
outlier_df[outlier_df['Summary'] >= 3]


# # Save outliers

# In[32]:


# Save outliers:
lst = list(outlier_df[outlier_df['Z Score (Conditional)'] == True].index)

import pickle
with open("Detected Outliers/HNC - Combined", "wb") as fp:   #Pickling
    pickle.dump(lst, fp)


# In[34]:


# Read outliers to check that it saved OK
with open("Detected Outliers/HNC - Combined", "rb") as fp:   # Unpickling
    b = pickle.load(fp)
    
b


# In[35]:


# Read all fresh csv files:
clinical = pd.read_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/clinical.csv', index_col='ID').drop(lst)
pet = pd.read_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/pet.csv', index_col='ID').drop(lst)
response = pd.read_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/response.csv', index_col='ID').drop(lst)

# Save with RO [removed outliers] extension:
clinical.to_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/clinical_combined_RO.csv')
pet.to_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/pet_combined_RO.csv')
response.to_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/response_combined_RO.csv')


# In[36]:


df.loc[lst]


# In[37]:


outlier_df.sum()


# In[ ]:




