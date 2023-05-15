#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


# Set path
import os
import sys

# Change this to the library for the utils folder on your computer
sys.path.insert(0, os.path.abspath('/Users/mikkorekstad/Skole/Master/paaskeProgging/Data Preparation/utils'))

# Basic python tools
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Formatting tools
# import formatting_tools

# Sklearn preprocessing tools
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# Other Computation Tools

# Plotting and visualization etc.
import matplotlib.pyplot as plt
import seaborn as sns


# # Combined data set

# In[2]:


# Read data from original file
my_dist = 1.1
clinical = pd.read_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/clinical_combined_RO.csv', index_col='ID')
pet = pd.read_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/pet_combined_RO.csv', index_col='ID')
df = pd.concat([clinical, pet], axis=1)

# Quick overview of the data
df.head()


# # Handling Multi-Colinearity
# - In this data set, there is a high amount of colinear features.model due to this issue.
# - Some of the features can be explained as combinations of the others.
# - This is especially an issue with some of the survival models. In fact, we struggled with fitting a CoxPH model to this data before handling the multi-colinearity problem.
# 
# ## hierarchical clustering on the Spearman rank-order correlations

# In[3]:


# First we create a copy and min-max scale it, this because the clustering algorithm is distance based
df_colinearity = df.copy()
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(df_colinearity)
for col, values in zip(df_colinearity.columns, X.T): df_colinearity[col] = values


# In[4]:


from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(X).correlation

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(
    dist_linkage, labels=list(df_colinearity.columns), ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro["ivl"]))

ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
fig.tight_layout()
plt.show()


# In[5]:


fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
corr = spearmanr(X).correlation

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(
    dist_linkage, labels=list(df_colinearity.columns), ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro["ivl"]))

#ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
#ax2.set_xticks(dendro_idx)
#ax2.set_yticks(dendro_idx)
#ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
#ax2.set_yticklabels(dendro["ivl"])
#fig.tight_layout()
ax1.set_ylabel('Distance [Ward Linkage]')
plt.title('hierarchical clustering on the Spearman rank-order correlations')
plt.show()


# In[6]:


fig, ax2 = plt.subplots(1, 1, figsize=(6, 4))
pos = ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]], vmin=-1, vmax=1)
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
#fig.tight_layout()
ax2.set_title('Heatmap of HNC Feature Correlation')


cbar = fig.colorbar(pos, ax=ax2)
#cax = plt.axes([0.92, 0.09, 0.04, 0.8])
#cbar = plt.colorbar(cax=cax)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Correlation', rotation=270)


plt.show()


# In[7]:


from collections import defaultdict

# Select distance based on visual inspection of plot above. Increase for lower amount of features.
# Can change also based on desired amount of features.

cluster_ids = hierarchy.fcluster(dist_linkage, my_dist, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
    
# Get indices
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
# Get feature names
selected_names = [df_colinearity.columns[i] for i in selected_features]
selected_names


# In[8]:


# Here is the complete overview of the clusters and their contents
for cluster, features in cluster_id_to_feature_ids.items():
    f_names = [df_colinearity.columns[i] for i in features]
    print(f'Cluster {cluster}: {f_names}')


# From this analysis we can see that all the PET - parameters are in the same cluster.
# Because of this, it might not be that interesting to compare the two data subsets separately.

# # Save results

# In[9]:


# Save the suggested features for later use
import pickle
with open('Feature Subsets/HeadNeckFeatures.pkl', "wb") as fp:   #Pickling
    pickle.dump(selected_names, fp)

