#!/usr/bin/env python
# coding: utf-8

# # Finding appropriate parametric models
# - Code from: https://lifelines.readthedocs.io/en/latest/Examples.html

# In[1]:


# Imports
from lifelines import *
from lifelines.plotting import qq_plot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ## OxyTarget

# In[2]:


response = pd.read_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/oxytarget/oxytarget_RO_response.csv', index_col='ID')

# Target columns
time = 'Time until OS event'
event = 'OS event'

T = response[time]
E = response[event]


# In[3]:


from lifelines.utils import find_best_parametric_model
best_model, best_aic_ = find_best_parametric_model(event_times=T,
                                                      event_observed=E,
                                                      scoring_method="AIC")


# In[4]:


best_model


# In[5]:


best_aic_


# In[6]:


response.describe()


# In[7]:


plt.clf()
sns.set_style('white')
sns.set_context("paper", font_scale = 2)
sns.displot(data=response, x=time, kind="hist", bins = 25, aspect = 1.5, hue=event, multiple="stack")
plt.show()


# In[8]:


plt.clf()
kmf = KaplanMeierFitter()
kmf.fit(T, event_observed=E)
kmf.survival_function_.plot()
plt.grid()
plt.title('Kaplan Meier Estimate OxyTarget')
plt.show()


# In[9]:


plt.clf()
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes = axes.reshape(4,)
models = [WeibullFitter(), LogNormalFitter(), LogLogisticFitter(), ExponentialFitter()]
model_names = ['Weibull', 'LogNormal', 'LogLogistic', 'Exponential']

oxy_dict = {}
for i, model in enumerate(models):
    model.fit(T, E)
    qq_plot(model, ax=axes[i], grid=True)
    axes[i].grid()
    print(f'{model_names[i]}: Log Likelihood [{model.log_likelihood_:.1f}], AIC [{model.AIC_:.1f}]')
    oxy_dict[model_names[i]] = f'{model.AIC_:.1f}'
fig.suptitle('OxyTarget QQ-Plots', fontsize=16)
plt.tight_layout()
plt.show()


# ### Discussion
# - QQ-plot suggests that lognormal is the best fitting parametric distribution.
# - This is supported by LogNormal: Log Likelihood [-313.676], AIC [631.352]
#     - AIC lower is better
#     - Log Likelihood higher is better

# ## Head Neck

# In[10]:


response = pd.read_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/headneck/response.csv', index_col='ID')

# Target columns
time = 'OS'
event = 'event_OS'

T = response[time]
E = response[event]


# In[11]:


response.describe()


# In[12]:


plt.clf()
sns.set_style('white')
sns.set_context("paper", font_scale = 2)
sns.displot(data=response, x=time, kind="hist", bins = 25, aspect = 1.5, hue=event, multiple="stack")

plt.show()


# In[13]:


plt.clf()
kmf = KaplanMeierFitter()
kmf.fit(T, event_observed=E)
kmf.survival_function_.plot()
plt.grid()
plt.title('Kaplan Meier Estimate Head Neck')
plt.show()
# kmf.cumulative_density_.plot()


# In[14]:


type(models[i])


# In[15]:


plt.clf()
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
#plt.title()
axes = axes.reshape(4,)
models = [WeibullFitter(), LogNormalFitter(), LogLogisticFitter(), ExponentialFitter()]
model_names = ['Weibull', 'LogNormal', 'LogLogistic', 'Exponential']

hnc_dict = {}
for i, model in enumerate(models):
    model.fit(T, E)
    qq_plot(model, ax=axes[i], grid=True)
    axes[i].grid()
    
    print(f'{model_names[i]}: Log Likelihood [{model.log_likelihood_:.3f}], AIC [{model.AIC_:.3f}]')
    hnc_dict[model_names[i]] = f'{model.AIC_:.1f}'
fig.suptitle('Head Neck QQ-Plots', fontsize=16)
plt.tight_layout()
plt.show()


# ### Discussion
# - QQ-plot suggests that lognormal is the best fitting parametric distribution.
# - This is supported by LogNormal: LogNormal: Log Likelihood [-443.258], AIC [890.516]
#     - AIC lower is better
#     - Log Likelihood higher is better

# In[16]:


pd.DataFrame.from_records([oxy_dict, hnc_dict], index=['OxyTarget', 'HeadNeck'])


# In[ ]:




