#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import pandas as pd
import numpy as np
import pickle
import warnings
# warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter, AalenAdditiveFitter, WeibullAFTFitter, LogNormalAFTFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
import copy
from matplotlib import pyplot as plt
import scipy.stats as st
# UTIL
from sksurv.util import Surv

# Metrics
from sksurv.metrics import concordance_index_censored as c_index
from sksurv.metrics import concordance_index_ipcw as uno_c
from sksurv.metrics import brier_score as brier
from sksurv.metrics import integrated_brier_score as ibs_score


# # Preparations: 

# In[2]:


# Load data
X_df = pd.read_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/oxytarget/oxytarget_RO.csv', index_col='ID')
response = pd.read_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/oxytarget/oxytarget_RO_response.csv', index_col='ID')

# Target columns
time = 'Time until OS event'
event = 'OS event'

# Folds loc
folds_loc = 'Folds/oxytarget.pkl'

# Subset loc
subset_loc = '../Exploratory Data Analysis/Feature Subsets/OxyTargetFeatures.pkl'
# subset_loc = '../Exploratory Data Analysis/Feature Subsets/OxyTargetFeaturesCherryPicked.pkl'

# Power Transform Yes/No
perform_power_transformation = True

# Binary encode continuous variables
binary = False


# # Load data
# X_df = pd.read_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/oxytarget/oxytarget_no_blood_samples_RO.csv', index_col='ID')
# response = pd.read_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/oxytarget/oxytarget_no_blood_samples_RO_response.csv', index_col='ID')
# 
# # Target columns
# time = 'Time until OS event'
# event = 'OS event'
# 
# # Folds loc
# folds_loc = 'Folds/oxytarget_no_blood_tests.pkl'

# # Load Data:

# In[3]:


# Load file with folds
with open(folds_loc, 'rb') as handle:
    folds = pickle.load(handle)

# Separate train and test folds
train_folds = folds['train']
test_folds = folds['test']

# Define fold names
fold_names = list(train_folds.keys())

# Load suggested feature subset
if subset_loc:
    with open(subset_loc, "rb") as openfile:
        subset = pickle.load(openfile)
        
        if 'Height (cm)' in subset:
            subset.remove('Height (cm)')
else:
    subset = None
    
if subset:
    X_df = X_df[subset]


# In[4]:


if binary:
    continuous = X_df[X_df.columns[X_df.nunique() > 1]].columns
    
    for col in continuous:
        X_df[col] = np.where(X_df[col] > X_df[col].median(), 1, 0) # 1 if over median


# # Define helping function

# In[5]:


def power_transform(X_train, X_test):
    
    # Define continous columns
    continous_cols = X_train.loc[:, (X_train.nunique() > 2)].columns
    
    # Fit a powertansformer using yeo-johnson
    transformer = PowerTransformer(method='yeo-johnson')
    transformer.fit(X_train[continous_cols])
    
    # Transform data
    X_train[continous_cols] = transformer.transform(X_train[continous_cols])
    X_test[continous_cols] = transformer.transform(X_test[continous_cols])
    
    # Return results
    return X_train, X_test


# In[ ]:





# # Define Utility Functions [Move to utils document later]

# In[6]:


# Train Surv model
def surv_model(model, X_train, Y_train, X_test, Y_test, time=time, event=event):
    
    # Powertransform if chosen:
    if perform_power_transformation:
        X_train, X_test = power_transform(X_train, X_test)
        
    # Define SK-Survival format DataFrames
    Y_train_sk = Y_train.copy()
    Y_test_sk = Y_test.copy()
    Y_train_sk[event] = [bool(val) for val in Y_train[event].values]
    Y_test_sk[event] = [bool(val) for val in Y_test[event].values]
    
    # Define combination of XY train
    X_Y_train = pd.concat([X_train, Y_train], axis=1)
    
    # Define highest encountered survival time
    y_max = max(Y_train[time].values)
    # times = np.arange(min(Y_test[time].values), max(Y_test[time].values))
    times = np.arange(12, 61)
    
    # Kaplan Meier
    if type(model) == KaplanMeierFitter:
        model.fit(Y_train[time], Y_train[event])
        pred = min(model.median_survival_time_, 400)
        c_y_pred_train = [pred for _ in range(len(X_train))]
        c_y_pred_test = [pred for _ in range(len(X_test))]
        survival_functions = [model.survival_function_at_times(times=times) for _ in range(len(X_test))]
        
    else:
        model.fit(X_Y_train, duration_col=time, event_col=event)
        
        # CoxPH use partial hazard for predictions
        if type(model) == CoxPHFitter:
            c_y_pred_train = model.predict_partial_hazard(X_train).values
            c_y_pred_test = model.predict_partial_hazard(X_test).values
            
        else: # For other models we use predict median
            c_y_pred_train = -np.clip(model.predict_median(X_train).values, 0, y_max)
            c_y_pred_test = -np.clip(model.predict_median(X_test).values, 0, y_max)
            
        
        survival_functions = model.predict_survival_function(X_test, times=times).T.values
        
        # Aalen can not estimate all survival times
        if type(model) == AalenAdditiveFitter:
            aaf_min = times[0]
            aaf_max = times[-1]
            aaf_times = np.array(model.predict_survival_function(X_train).index)
            
            # Create filter for finding correct survival times:
            predictions = model.predict_survival_function(X_test, times=times)
            _filter = ((predictions.index >= aaf_min) &
                      (predictions.index <= aaf_max))
            
            # Apply
            predictions = predictions[_filter]
            survival_functions = predictions.T.values
            times = aaf_times[(aaf_times >= aaf_min) & (aaf_times <= aaf_max)]
            # times = [t for t in  if t >= aaf_min and t <= aaf_max]
        
            
    # Calculate C-index
    c_index_train = c_index(Y_train_sk[event], Y_train_sk[time], c_y_pred_train)[0]
    c_index_test = c_index(Y_test_sk[event], Y_test_sk[time], c_y_pred_test)[0]
    
    # Calculate Uno's C-index
    tau = response[time][response[event] == 1].max()
    Y_train_sksurv = Surv.from_arrays(event=Y_train_sk[event], time=Y_train_sk[time])
    Y_test_sksurv = Surv.from_arrays(event=Y_test_sk[event], time=Y_test_sk[time])
    uno_c_score = uno_c(Y_train_sksurv, Y_test_sksurv, c_y_pred_test, tau=tau)[0]
    
    # Calculate Integrated Brier score
    # ibs_times = np.arange(min(Y_test[time].values) + 1, max(Y_test[time].values))
    # ibs_times = np.arange(12, 61, 1)
    
    ibs = ibs_score(Y_train_sksurv, Y_test_sksurv, survival_functions, times=times)
    
    regular_brier = brier(Y_train_sksurv, Y_test_sksurv, survival_functions, times=times)
    
    return {'model': model, 
            'c_index_train': c_index_train, 'c_index_test': c_index_test, 
            'uno_c': uno_c_score, 'ibs': ibs, 'brier': regular_brier
           }# , 'uno-c': uno_c, 'ibs': ibs, 
            #'train_c-index': c_index_train, 'train_uno-c': uno_c_train, 'train_ibs': ibs_train}


# ## Train models

# In[7]:


results = {}

for fold_name in fold_names: # [0:1]
    
    
    # Print progress update
    if int(fold_name[-1]) % 5 == 0:
        printing = True
        print(f'{fold_name}')
    
    # Train and Test IDs / Names:
    train_names = train_folds[fold_name]
    test_names = test_folds[fold_name]
    
    # Train and Test DataFrames
    X_train = X_df.loc[train_names]
    X_test = X_df.loc[test_names]
    Y_train = response.loc[train_names][[time, event]]
    Y_test = response.loc[test_names][[time, event]]
    X_Y_train = pd.concat([X_train, Y_train], axis=1) # Lifelines needs X and Y in the same DF for training
    X_Y_test = pd.concat([X_test, Y_test], axis=1) # Probably don't need it for test, but just in case
    
    surv_models = {
        # 'KM': {'model': KaplanMeierFitter()},
        # 'CPHL1': {'model': CoxPHFitter(penalizer=0.1, l1_ratio=1.0)}, 
        'CPH': {'model': CoxPHFitter(penalizer=0.1, l1_ratio=0.0)},
        'AAF': {'model': AalenAdditiveFitter(coef_penalizer=0.1)}, 
        'WAFT': {'model': WeibullAFTFitter(penalizer=0.1, l1_ratio=0.0)},
        'LNAFT': {'model': LogNormalAFTFitter(penalizer=0.1, l1_ratio=0.0)}
    }

    for model_name, model_content in surv_models.items():
        surv_models[model_name] = surv_model(model_content['model'], X_train, Y_train, X_test, Y_test)
    
    results[fold_name] = surv_models


# # Get results
# for fold_name, fold_content in results.items():
#     for model_name, model_content in fold_content.items():
#         inverted_results[model_name][fold_name] = model_content

# In[8]:


metrics = ['c_index_train', 'c_index_test', 'uno_c', 'ibs']
models = list(results['fold_1'].keys())
fold_names = list(results.keys())
inverted_results = {model: {fold: results[fold][model] for fold in results.keys()} for model in models}
recorded_metrics = {model: {} for model in models}
recorded_metrics_mean = {model: {} for model in models}
recorded_metrics_str = {model: {} for model in models}

for model_name, model_content in inverted_results.items():
    print(f'{model_name}:')
    for metric in metrics:
        
        # Collect the scores
        metric_scores = [inverted_results[model_name][fold][metric] for fold in fold_names]
        
        # List of metrics
        recorded_metrics[model_name][metric] = metric_scores
        
        # Mean
        recorded_metrics_mean[model_name][metric] = np.mean(metric_scores)
        
        # Nice string
        nice_str = f'{np.mean(metric_scores):.3f} +/- {np.std(metric_scores):.3f}'
        recorded_metrics_str[model_name][metric] = nice_str
        print(f'{metric}: {nice_str}')


# In[9]:


res = pd.DataFrame.from_dict(recorded_metrics_str)
res


# In[10]:


# res.drop(['KM', 'CPHL1'], axis=1)


# In[11]:


models = ['CPH', 'AAF', 'WAFT', 'LNAFT']
brier_scores = {}#aaf_brier = []

for model in models:
    
    lst = []
    for fold in inverted_results[model]:
        timeline = scores = inverted_results[model][fold]['brier'][0]
        scores = inverted_results[model][fold]['brier'][1]
        lst.append(pd.Series(scores, index=timeline))
        
    brier_scores[model] = lst


# In[12]:


brier_dct = {}
for name, scores in brier_scores.items():#name, scores in zip(names, scores_list):
    m = pd.DataFrame.from_records(scores).mean()
    cl = m - 1.96 * pd.DataFrame.from_records(scores).sem()# mean()
    cu = m + 1.96 * pd.DataFrame.from_records(scores).sem()

    res = pd.DataFrame([m, cl, cu]).T
    res.columns = ['mean', 'CI lower', 'CI upper']
    #res.index = m.index
    
    brier_dct[name] = res


# In[13]:


# occurrence_rate


# In[28]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))#,# sharex=True, sharey=True, fig)

occurrence_rate = response[event].mean()
non_informative =  occurrence_rate * (1 - occurrence_rate) ** 2 + (1 - occurrence_rate) * occurrence_rate ** 2

for ax, model in zip(axes.flatten(), brier_dct):
    # model = inverted_results[model_name]['fold_1']['model']
    # survival_probability_calibration(model, surv_test, t0=pred_t, ax=ax)
    # model.plot_partial_effects_on_outcome(covariates, values, ax=ax)
    time_line = brier_dct[model].index
    
    mean_scores = brier_dct[model]['mean']
    ci_lower = brier_dct[model]['CI lower']
    ci_upper = brier_dct[model]['CI upper']
    
    ax.plot(time_line, mean_scores)
    ax.fill_between(time_line, ci_lower, ci_upper, color='b', alpha=.1)
    ax.set_title(f'{model}')
    ax.set_xlabel('Overall survival time [months]')
    ax.set_ylabel('Brier Score')
    ax.set_yticks(np.arange(0.05, 0.26, 0.05))
    ax.set_xticks(np.arange(12, 61, 12))
    ax.set_yticks(np.arange(0.05, 0.26, 0.05))
    ax.set_ylim(0.05, 0.25)
    ax.set_xlim(12, 60)
    ax.hlines(0.25, xmin=12, xmax=60, colors='k', linestyles='--')
    ax.grid()
    
fig.suptitle(f'Brier scores dependent on time')
plt.tight_layout()


# In[ ]:





# In[30]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd

from sklearn import set_config
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sksurv.datasets import load_flchain, load_gbsg2
from sksurv.functions import StepFunction
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.preprocessing import OneHotEncoder, encode_categorical
from sksurv.util import Surv


# In[31]:


gbsg_X, gbsg_y = load_gbsg2()


# In[38]:


event = [1 for lst in gbsg_y if lst[0] == 1]


# In[41]:


sum(event) / len(gbsg_y)


# In[ ]:





# 

# In[15]:


results['fold_1']['WAFT']['model'].check_assumptions


# In[16]:


def plot_survival_function(model, values, covariates, baseline=False):

    for value in values:
        sample = pd.DataFrame(np.zeros(len(X_df.columns))).T
        sample.columns = X_df.columns
        
        label = ''
        for i, covariate in enumerate(covariates):
            sample[covariate] = value[i]
            label += covariate + '=' + str(value)
        model.predict_survival_function(sample).plot(ax=ax, label=label)
    
    if baseline:
        sample = pd.DataFrame(np.zeros(len(X_df.columns))).T
        sample.columns = X_df.columns
        model.predict_survival_function(sample).plot(ax=ax, label='baseline', ls=":", color="k")


# In[17]:


covariates = ['Suspected metastatic lesions at diagnosis', 'Mucinous']#, 'Mucinous']
values = [[0, 1], [1, 0]]

cali_models = ['CPH', 'AAF','WAFT', 'LNAFT']
num = [1, 2, 3, 4]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))#,# sharex=True, sharey=True, fig)

for ax, model_name, num in zip(axes.flatten(), cali_models, num):
    model = inverted_results[model_name]['fold_1']['model']
    # survival_probability_calibration(model, surv_test, t0=pred_t, ax=ax)
    
    # plot_partial_effects_on_outcome(covariates=covariates, values=values, ax=ax, self=model)
    if isinstance(model, AalenAdditiveFitter):
        # model.plot(columns=covariates, ax=ax)
        plot_survival_function(model, values, covariates, baseline=True)
        ax.set_title(f'Survival Function {model_name}')
        ax.legend()
        
        
    else:
        model.plot_partial_effects_on_outcome(covariates, values, ax=ax, legend=False)
        ax.set_title(f'Partial effect on outcome {model_name}')
    
    ax.set_xlabel('Time in months')

h, l = axes.flatten()[-1].get_legend_handles_labels()
fig.legend(h, labels=l,
           loc="upper right",
          bbox_to_anchor=(1.05, 0.92))


[ax.get_legend().remove() for ax in axes.flatten()]
 
fig.suptitle(f'HNC: Demonstration of effect from  {covariates}')
# plt.subplots_adjust(right=0.8)
plt.tight_layout()


# # Observed vs. Predicted survival times

# In[18]:


train = train_folds['fold_1']
test = test_folds['fold_1']

X_train = X_df.loc[train]
X_test = X_df.loc[test]

if perform_power_transformation:
        X_train, X_test = power_transform(X_train, X_test)

Y_train = response[[time, event]].loc[train]
Y_test = response[[time, event]].loc[test]

surv_train = pd.concat([X_train, Y_train], axis=1)
surv_test = pd.concat([X_test, Y_test], axis=1)


# In[19]:


from lifelines.utils import CensoringType
from lifelines.fitters import RegressionFitter
from lifelines import CRCSplineFitter
def survival_probability_calibration2(model, df, t0, ax=None):
    r"""
    Smoothed calibration curves for time-to-event models. This is analogous to
    calibration curves for classification models, extended to handle survival probabilities
    and censoring. Produces a matplotlib figure and some metrics.
    We want to calibrate our model's prediction of :math:`P(T < \text{t0})` against the observed frequencies.
    Parameters
    -------------
    model:
        a fitted lifelines regression model to be evaluated
    df: DataFrame
        a DataFrame - if equal to the training data, then this is an in-sample calibration. Could also be an out-of-sample
        dataset.
    t0: float
        the time to evaluate the probability of event occurring prior at.
    Returns
    ----------
    ax:
        mpl axes
    ICI:
        mean absolute difference between predicted and observed
    E50:
        median absolute difference between predicted and observed
    https://onlinelibrary.wiley.com/doi/full/10.1002/sim.8570
    """

    def ccl(p):
        return np.log(-np.log(1 - p))

    if ax is None:
        ax = plt.gca()

    T = model.duration_col
    E = model.event_col

    # Make the change her: Predicted at times is not supported yet in lifelines
    # So we take the nearest point in time, which is approximately 59.2
    pred = model.predict_survival_function(df)
    # pred.index = np.round(pred.index)
    # print(pred.index)
    # pred = pred.loc[59.0]
    # print(pred)
    # closest_index = pred.index.sub(t0).abs().idxmin()
    # print(closest_index)
    pred_timeline = list(pred.index)
    closest = pred_timeline[min(range(len(pred_timeline)), key = lambda i: abs(pred_timeline[i]-t0))]
    pred = pred.loc[closest]
    pred = pd.DataFrame(pred).T
    pred.index = [t0]
    
    # From here everything is as usual
    predictions_at_t0 = np.clip(1 - pred.T.squeeze(), 1e-10, 1 - 1e-10)

    # create new dataset with the predictions
    prediction_df = pd.DataFrame({"ccl_at_%d" % t0: ccl(predictions_at_t0), T: df[T], E: df[E]})

    # fit new dataset to flexible spline model
    # this new model connects prediction probabilities and actual survival. It should be very flexible, almost to the point of overfitting. It's goal is just to smooth out the data!
    n_knots = 3
    regressors = {"beta_": ["ccl_at_%d" % t0], "gamma0_": "1", "gamma1_": "1", "gamma2_": "1"}

    # this model is from examples/royson_crowther_clements_splines.py
    crc = CRCSplineFitter(n_baseline_knots=n_knots, penalizer=0.000001)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if CensoringType.is_right_censoring(model):
            crc.fit_right_censoring(prediction_df, T, E, regressors=regressors)
        elif CensoringType.is_left_censoring(model):
            crc.fit_left_censoring(prediction_df, T, E, regressors=regressors)
        elif CensoringType.is_interval_censoring(model):
            crc.fit_interval_censoring(prediction_df, T, E, regressors=regressors)

    # predict new model at values 0 to 1, but remember to ccl it!
    x = np.linspace(np.clip(predictions_at_t0.min() - 0.01, 0, 1), np.clip(predictions_at_t0.max() + 0.01, 0, 1), 100)
    y = 1 - crc.predict_survival_function(pd.DataFrame({"ccl_at_%d" % t0: ccl(x)}), times=[t0]).T.squeeze()

    # plot our results
    ax.set_title("Smoothed calibration curve of \npredicted vs observed probabilities of t ≤ %d mortality" % t0)

    color = "tab:red"
    ax.plot(x, y, label="smoothed calibration curve", color=color)
    ax.set_xlabel("Predicted probability of \nt ≤ %d mortality" % t0)
    ax.set_ylabel("Observed probability of \nt ≤ %d mortality" % t0, color=color)
    ax.tick_params(axis="y", labelcolor=color)

    # plot x=y line
    ax.plot(x, x, c="k", ls="--")
    ax.legend()

    # plot histogram of our original predictions
    color = "tab:blue"
    twin_ax = ax.twinx()
    twin_ax.set_ylabel("Count of \npredicted probabilities", color=color)  # we already handled the x-label with ax1
    twin_ax.tick_params(axis="y", labelcolor=color)
    twin_ax.hist(predictions_at_t0, alpha=0.3, bins="sqrt", color=color)

    plt.tight_layout()

    deltas = ((1 - crc.predict_survival_function(prediction_df, times=[t0])).T.squeeze() - predictions_at_t0).abs()
    ICI = deltas.mean()
    E50 = np.percentile(deltas, 50)
    print("ICI = ", ICI)
    print("E50 = ", E50)

    return ax, ICI, E50


# In[20]:


from lifelines.calibration import survival_probability_calibration
pred_t = 60
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))#,# sharex=True, sharey=True, fig)

models = ['CPH', 'AAF', 'WAFT', 'LNAFT']

for ax, model_name in zip(axes.flatten(), models):
    model = inverted_results[model_name]['fold_1']['model']
    
    if isinstance(model, AalenAdditiveFitter):
        survival_probability_calibration2(model, surv_test, t0=pred_t, ax=ax)
        # continue
    else:
        survival_probability_calibration(model, surv_test, t0=pred_t, ax=ax)
    ax.set_title(f'{model_name}')
    
fig.suptitle(f'Smoothed calibration curve of predicted vs. observed probabilities of t $\leq$ {pred_t}')
plt.tight_layout()


# # Assumptions 

# In[21]:


surv_df = pd.concat([X_df, response[[time, event]]], axis=1)
cph = CoxPHFitter(penalizer=0.1, l1_ratio=0)
cph.fit(surv_df, duration_col=time, event_col=event)
cph.check_assumptions(surv_df, show_plots=True, p_value_threshold=0.05)

