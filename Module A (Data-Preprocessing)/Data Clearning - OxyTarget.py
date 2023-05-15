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
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


# # Loading Data

# In[2]:


include_blood = True

# Read data from original file
df = pd.read_excel('/Users/mikkorekstad/Skole/master_data/raw_data/oxytarget/201119_OxyTarget_datasheet.xlsx', index_col='ID')
n_rows = df.shape[0]
n_cols = df.shape[1]
print(f'Read df in with {n_rows} rows and {n_cols} columns.')

# Now remove rows with withdrawn consent
df= df[df['Date of inclusion'] != 'Withdrawn consent']
df = df[df['OS event'].notna()]
n_rows = df.shape[0]
n_cols = df.shape[1]
print(f'Reduced to {n_rows} rows and {n_cols} columns.')
df.head()


# # Clean Target Columns
# - For a section of the data cleaning we remove the target from the data set, so that no information leaks.

# In[3]:


df['PFS-event'].sum()


# In[4]:


# Recalculate duration columns
# Date of inclusion or Date MR w/diagnose
df['Time until OS event'] = (df['Last registered alive'] - df['Date MR w/diagnose']).dt.days // 30
df['Time until PFS event'] = np.where(df['PFS-event'] == 1, df['Time until PFS event'], 
                                      (df['Last registered alive'] - df['Date MR w/diagnose']).dt.days // 30)

# Extract and remove target columns
targets = df[['PFS-event', 'Time until PFS event', 'OS event', 'Time until OS event']]
df.drop(['PFS-event', 'Time until PFS event', 'OS event', 'Time until OS event'], axis=1, inplace=True)


# In[5]:


df['diff'] = (df['Date MR w/diagnose'] - df['Date of inclusion']).dt.days // 30
df['diff'].value_counts()


# In[6]:


# Patient 160 was included in the study exactly 1 year before being included in diagnosed wiht MR.
df[df['diff'] == 12] # This patient is censored, so if there is an error, we only get reduced info, not incorrect info


# In[7]:


# Patient 52 was included in the study exactly 2 months before being included in diagnosed wiht MR.
df[df['diff'] == 2] # The same goes for patient 52


# In[8]:


df.drop(['diff'], axis=1, inplace = True)


# In[9]:


df[['Sex (M/F)', 'MSI', 'TRG (CAP/AJCC)', 'TRG (Bateman)']].isnull().sum()


# In[10]:


# Change '-' and similar to NaN
df.replace({'-': np.nan, 'nf': np.nan, 'NF': np.nan, '?': np.nan, 
            '                                                                        ':np.nan}, 
           inplace=True)


# # Inspect surgery column

# In[11]:


print(f"Patients with no surgery: {len(df[df['Date surgery'] == 'No'])}")# .isnull().sum()
# surgery_dates = df['Date surgery']
temp_lst = df[['Date of inclusion', 'Date surgery', 'Type of surgery']][df['Type of surgery'].isnull() == True].index
df[['Date of inclusion', 'Date surgery', 'Type of surgery', 'MORS']][df['Type of surgery'].isnull() == True]


# In[12]:


df['Type of surgery'] = np.where(df['Date surgery'] == 'No', 'No', df['Type of surgery'])
df.loc[temp_lst][['Date of inclusion', 'Date surgery', 'Type of surgery']]


# # Inspect Observations with many missing data points
# - Observations with too many missing data points can be problematic. In this subsection we take a look at the worst cases.

# In[13]:


print(f'Average missing value count: {df.T.isnull().sum().sort_values().mean():.2f}')
print(f'Minimum missing value count: {df.T.isnull().sum().sort_values().min():.2f}')
print(f'Max missing value count: {df.T.isnull().sum().sort_values().max():.2f}')
df.T.isnull().sum().sort_values()[-10:]


# In[14]:


df.loc['OxyTarget 070']


# # Removal of Columns

# ## Unecessary and / or difficult to process information

# In[15]:


# Remove Date-Columns
dates = ['Date of inclusion', # B
         'Date of referral to specialist', # J
         'Date primary biopsy', # M
         'Date MR w/diagnose', # T
         'Date chemotherapy', # AG
         'Date start RT', # AI
         'Date RT finish', # AK 
         'Date post-CRT MR', # AL
         'Date other radiology', # AD
         'Date metastatic disease', # CJ PSF ? 
         'Date local recurrence', # CL
         'MORS', # CN
         'Last registered alive', # CQ 
         'Date surgery' # AN
        ]

# Other columns to remove upfront
other_to_remove = ['Symptoms', 'Description post-CRT MR', 'Comment', # Too many different formats
                   'Other cancer', 'Comment adjuvant treatment', 'Other exams', 'Comments pathology', # --.--
                   # 'Sex (M/F)', 'MSI', 'TRG (CAP/AJCC)', 'TRG (Bateman)', # Already encoded these
                   'Further follow up', 'Included in other study', 'Biopsy histology ID', # Non-informative
                   'Histology reference no.', 'Place Surgery' # --.--
                    ]

df.drop(dates + other_to_remove, axis=1, inplace=True)
print(f'Reduced to {df.shape[1]} columns.')


# In[16]:


blood_test = ['CEA baseline', 'CRP baseline', 'Hemoglobin (g/dl)',
       'ESR (mm/h)', 'Thrombocytes (10ˆ9/L)', 'Leukocytes (10ˆ9/L)',
       'Neutrophils (10ˆ9/L)', 'Lymphocytes (10ˆ9/L)', 'Monocytes (10ˆ9/L)',
       'Eosinophils (10ˆ9/L)', 'Basophils (10ˆ9/L)', 'Albumin (g/L)',
       'Sodium (mmol/L)', 'Potassium (mmol/L)', 'Calcium (total) (mmol/L)',
       'Chloride (mmol/L)', 'Creatinine (umol/L)', 'Carbamide',
       'Bilirubin (umol/L)', 'ALT (U/L)', 'AST (U/L)', 'GT (U/L)', 'ALP (U/L)',
       'LDH (U/L)', 'Blood type']

if not include_blood:
    # If we want to, it is easy now to drop the blood test features
    for col in blood_test:
        try:
            df.drop([col], axis=1, inplace=True)
        except:
            print(f'Had issues removing {col}')
    print(f'Reduced to {df.shape[1]} columns.')


# ## Columns with too much missing information

# In[17]:


df.isnull().sum()


# In[18]:


# Treshold 
threshold = 0.25

# Remove the columns below threshold
df = df.loc[:, df.isna().mean() < threshold]
print(f'Reduced to {df.shape[1]} columns.')


# In[19]:


df, numeric, numeric_counts = formatting_tools.format_numeric(df)


# In[20]:


# Fix troublesome input value
try:
    formatted_values = []
    for value in df['Hemoglobin (g/dl)']:
        if ',' in str(value):
            print(value)
            value = str(value).replace(',', '.')
        formatted_values.append(value)

    df['Hemoglobin (g/dl)'] = formatted_values
except:
    print('Probably already removed Hemoglobin (g/dl)')


# In[21]:


# Taking a new look at missing values
print(f'Average missing value count: {df.T.isnull().sum().sort_values().mean():.2f}')
print(f'Minimum missing value count: {df.T.isnull().sum().sort_values().min():.2f}')
print(f'Max missing value count: {df.T.isnull().sum().sort_values().max():.2f}')
df.T.isnull().sum().sort_values()[-10:]


# The missing value counts looks a lot better now, with the average missing value count at 2.05. The maximum samples has around 1 / 3 missing data points.

# # Encoding Categories

# In[22]:


# numeric.append('Hb baseline')
# Start with finding the non-numeric columns
non_numeric = [feature for feature in list(df.columns) if feature not in numeric]
non_numeric


# In[23]:


# Correct some minor differences in the data-sheet
repl_dict = {'Histology description': {'Adenocarcinoma   ': 'Adenocarcinoma', 
                                       'Adenocarcinoma ': 'Adenocarcinoma',
                                       'Fibrose': 'Fibrosis'},
             'Differentiation' :{'Hight': 'High', 'Moderat': 'High'},
             'Location primary tumor': {'Rectum          ': 'Rectum'}
    }      
df = df.replace(repl_dict)


# In[24]:


formatting_tools.print_category_distributions(df, non_numeric)


# ## Encoding Non-Numerical Categories
# It looks like some of the categories have very few patients. This also includes missing values.
# Now we have to make a decision on each feature and category, on how to numerically represent the feature.
#     
# - Binary encode: Location primary tumor [0 missing values]
#     - Almost 86% of samples in category "Rectum". Other categories maximum 6%. We binary encode this feature to be Rectum 1, other 0.
# 
# - Binary encode and change name: Sex (M/F) [0 missing values]
#     - Change name to fit style of other data set. Female = 1, male = 0
#     
# - Binary encode: Cancer type [0 missing values]
#     - The same as with "location primary tumor", "Adenocarcinoma" = 1, other = 0.
#     
# - Binary encode: Suspected metastatic lesions at diagnosis [0 missing values]
#     - This feature already is binary. "Yes" = 1, other = 0.
#     
# - Binary encode: Neoadjuvant CRT (Yes/No) [0 missing values]
#     - This feature already is binary. "Yes" = 1, other = 0.
#     
# 
# - Binary encode: Adjuvant treatment [0 missing values]
#     - This feature already is binary. "Yes" = 1, other = 0.
#     
# 
# ### Features with missing values:
#     - Missing values are imputed using the most frequent method before separed into binary columns.
#     
# - DROP: Blood samples at inclusion [1 missing values]
#     - Almost all samples have "yes" category. Will not give us information.
#     - NaN values does not matter because we drop this column all together.
#     
# - DROP: MSI [19 missing values]
#     - Non-informative: Almost all values are the same!
#     
# - Split into Three binary columns: Type of surgery [27 missing values]
#     - This feature have three categories with more than 10% of the population. In order to cut down on some features, we encode this feature into three binary columns, one for each of these three categories. LAR [41.77%], APR [28.48%], UAR [13.92%]. For these features, 1 = if this is the surgery, 0 for other. In this matter, if all three columns are 0, it is one of the other surgeries.
#     - Because of the number of missing values in type of surgery is equal to the number of patients not having surgery, we assume that the missing values are the patients with no surgery. The missing values is therefore left included in the 'Other' categories. We might introduce a feature column for patients with no surgery later.
#     
# - Binary encode: Histology description [34 missing values]
#     - 89.4% of samples in category "Adenocarcinoma". Other categories maximum 6.62 %. We binary encode this feature to be "Adenocarcinoma" 1, other 0.
#     - NaN values carried forward and imputed later.
# 
# - Two binary columns: Differentiation [46 missing values]
#     - One for low (1 = low, 0 other) and one for high (1 = high, 0 other). In this case, mean is 0 in both columns.
#     - For this column it is assumed that NaN values give some information. In this experiment, we therefore include this in the "other" together with medium differentiation.
#     
# - Binary encode: Mucinous [35 missing values]
#     - This feature already is binary. "Yes" = 1, other = 0.
#     
# 

# In[25]:


# Encode some of the suggestions from the information sheet to the data set
df['female'] = np.where(df['Sex (M/F)'] == 'F', 1, 0) # 1 if female, otherwise 0

# Other features with no missing values first:
df['Location primary tumor_Rectum'] = np.where(df['Location primary tumor'] == 'Rectum', 1, 0)
df['Cancer type_Adenocarcinoma'] = np.where(df['Cancer type'] == 'Adenocarcinoma', 1, 0)
df['Neoadjuvant CRT'] = np.where(df['Neoadjuvant CRT (Yes/No)'] == 'Yes', 1, 0)
df['Suspected metastatic lesions at diagnosis'] = np.where(df['Suspected metastatic lesions at diagnosis'] == 'Yes', 1, 0)
df['Adjuvant treatment'] = np.where(df['Adjuvant treatment'] == 'Yes', 1, 0)


# In[26]:


# Define function for handling missing values:
#df['Mucinous2'] = np.where(df['Mucinous'] == 'Ja', 1, 0)
def carry_nans(series, condition):
    backup = series
    series = condition
    return np.where(backup.isnull() == True, np.nan, series)


# In[27]:


surgery_other = np.where((df['Type of surgery'] == 'Transanal endoscopic microsurgery') # Too sparse
         | (df['Type of surgery'] == 'UAR') # Complete separation of OS - event
         | (df['Type of surgery'] == 'Polypectomy'), 1, 0) # Too sparse


blood_other = np.where((df['Blood type'] == 'A-') # Too sparse
         | (df['Blood type'] == '0-') # Too sparse
         | (df['Blood type'] == 'B-') # Too sparse
         | (df['Blood type'] == 'AB-') # Too sparse
         | (df['Blood type'] == 'AB+'), 1, 0) # Too sparse


# In[28]:


# Impute data
to_impute = ['Mucinous', 'Differentiation', 'Type of surgery', 'Histology description', 'Blood type']
impute_df = df[to_impute]
imuted_data = SimpleImputer(strategy='most_frequent').fit_transform(impute_df)
for col, values in zip(to_impute, imuted_data.T): df[col] = values


# In[29]:


# Simple binary columns
df['Mucinous'] = carry_nans(df['Mucinous'], np.where(df['Mucinous'] == 'Ja', 1, 0))

# Separated Differentiation into two columns, not three (drop first principle)
df['Differentiation_High'] = carry_nans(df['Differentiation'], np.where(df['Differentiation'] == 'High', 1, 0))
df['Differentiation_Low'] = carry_nans(df['Differentiation'], np.where(df['Differentiation'] == 'Low', 1, 0))

# Type of surgery into three columns, the other categories were too rare
df['Type of surgery_LAR'] = carry_nans(df['Type of surgery'], np.where(df['Type of surgery'] == 'LAR', 1, 0))
df['Type of surgery_APR'] = carry_nans(df['Type of surgery'], np.where(df['Type of surgery'] == 'APR', 1, 0))
# df['Type of surgery_UAR'] = carry_nans(df['Type of surgery'], np.where(df['Type of surgery'] == 'UAR', 1, 0))
df['Type of surgery_Hartmann'] = carry_nans(df['Type of surgery'], np.where(df['Type of surgery'] == 'Hartmann', 1, 0))
df['Type of surgery_Other'] = carry_nans(df['Type of surgery'], surgery_other)
df['No surgery'] = carry_nans(df['Type of surgery'], np.where(df['Type of surgery'] == 'No', 1, 0))

# Almost all 'Histology descriptions' are 'Adenocarcinoma', so turn it to binary
df['Histology description_Adenocarcinoma'] = carry_nans(df['Histology description'], np.where(df['Histology description'] == 'Adenocarcinoma', 1, 0))
# df['Histology description_Adenocarcinoma'] = formatting_tools.carry_nan(df, from_='Histology description', to_='Histology description_Adenocarcinoma')

if include_blood:
    df['Blood type_a_plus'] = carry_nans(df['Type of surgery'], np.where(df['Blood type'] == 'A+', 1, 0))
    df['Blood type_b_plus'] = carry_nans(df['Type of surgery'], np.where(df['Blood type'] == 'B+', 1, 0))
    df['Blood type_0_plus'] = carry_nans(df['Type of surgery'], np.where(df['Blood type'] == '0+', 1, 0))
    df['Blood type_other'] = carry_nans(df['Blood type'], blood_other)


# In[30]:


# Drop the now redundant columns
to_drop = ['Blood samples at inclusion', 'Neoadjuvant CRT (Yes/No)', 'Differentiation', 'Location primary tumor', 
           'Type of surgery', 'Cancer type', 'Histology description', 'Sex (M/F)', 'MSI']

if include_blood:
    to_drop.append('Blood type')

df.drop(to_drop, axis=1, inplace=True)
print(f'Now {df.shape[1]} columns.')


# In[31]:


# Store these features in a list in case it's needed later
cats = ['Adjuvant treatment', 'Mucinous', 'Neoadjuvant CRT', 'Suspected metastatic lesions at diagnosis', 
        'Differentiation_High', 'Differentiation_Low', 'Type of surgery_LAR', 'Type of surgery_APR', 'Type of surgery_UAR',
        'Histology description_Adenocarcinoma', 'Histology description_Adenocarcinoma', 'Cancer type_Adenocarcinoma',
        'Location primary tumor_Rectum'
       ]


# ## Numerical categories
# - First we set a threshold for what to call contiuous and what to call categorical.
# - Then we make some corrections on the variables that are wrongly formatted.
# 

# In[32]:


# Set a threshold for continuous variables.
continous_threshold = 10

# Confirmed numeric list
confirmed_numeric = []
categorical_with_numeric_values = []

# Iterate over list of unique counts for numerical features.
for col, count in numeric_counts.items():
    
    # Add column to list based on threshold criteria
    if count < continous_threshold:    
        categorical_with_numeric_values.append(col)
    else:
        confirmed_numeric.append(col)


# In[33]:


# Formatting
numeric_corrections = {
    'mrT    (TNM ed.7)': {'0': 0, '2': 2, '3': 3, '4': 4 },
    'p/ypT       (TNM ed.7)': {'4a': 4} # Encoding this to binary because of high amount of missing values
}

df.replace(numeric_corrections, inplace=True)

# Print overview
formatting_tools.print_category_distributions(df, categorical_with_numeric_values)


# ## Encoding numerical categories
# 
# - To binary: mrT    (TNM ed.7) [12 missing values]
#     - Similar to TRG, this is encoded binary. This is to reduce the amount of features.
#     - 3 and 4 is 1, other is 0. This includes NaN values. Assumes that if the values were 3-4 it would have been journaled.
#     
# - To Binary: mrN [5 missing values]
#     - 2-3 is 1, other is 0. This includes NaN values. Assumes that if the values were 2-3 it would have been journaled.
#     
# - mrV [0 missing values] is good as is.
# 
# - Stadium (ACR 2016) [11 missing values]
#     - 3-4 is 1, other is 0. Assumes that if the values were 3-4 it would have been journaled.
#     
# - p/ypT       (TNM ed.7) [39 missing values]
#     - 3-4 and 4a is 1, other is 0. Assumes that if the values were 3-4 or 4a it would have been journaled.
#     
# - p/ypN (TNM ed. 7) [42 missing values]
#     - 1-2 is 1, 0 and NaN is 0. Assumes that NaN is some information correlated with 0 in this case.
#     
# - R classification [30 missing values]:
#     - Left as is. NaN values will be imputed for this.

# In[34]:


# Impute data
to_impute = ['p/ypN (TNM ed. 7)', 'p/ypT       (TNM ed.7)', 'Stadium (ACR 2016)', 'mrN', 'mrT    (TNM ed.7)']
impute_df = df[to_impute]
imuted_data = SimpleImputer(strategy='most_frequent').fit_transform(impute_df)
for col, values in zip(to_impute, imuted_data.T): df[col] = values


# In[35]:


# Binary encode some of the features to get larger groups
df['p/ypN (TNM ed. 7)_1-2'] = carry_nans(df['p/ypN (TNM ed. 7)'], np.where(df['p/ypN (TNM ed. 7)'] >= 1, 1, 0))
df['p/ypT       (TNM ed.7)_3-4-4a'] = carry_nans(df['p/ypT       (TNM ed.7)'], np.where(df['p/ypT       (TNM ed.7)'] >= 3, 1, 0))
df['Stadium (ACR 2016)_3-4'] = carry_nans(df['Stadium (ACR 2016)'], np.where(df['Stadium (ACR 2016)'] >= 3, 1, 0))
df['mrN_2-3'] = carry_nans(df['mrN'], np.where(df['mrN'] >= 2, 1, 0))
df['mrT    (TNM ed.7)_3-4'] = carry_nans(df['mrT    (TNM ed.7)'], np.where(df['mrT    (TNM ed.7)'] >= 3, 1, 0))

# Remove the redundant columns
to_drop = ['p/ypN (TNM ed. 7)', 'p/ypT       (TNM ed.7)', 'Stadium (ACR 2016)', 'mrN', 'mrT    (TNM ed.7)']
df.drop(to_drop, axis=1, inplace=True)


# In[36]:


# Store these in a list
num_cats = ['p/ypN (TNM ed. 7)_1-2', 'p/ypT       (TNM ed.7)_3-4-4a', 'Stadium (ACR 2016)_3-4', 'mrN_2-3',
           'mrT    (TNM ed.7)_3-4', 'mrV', 'R classification']


# # Numerical Columns

# In[37]:


# Check missing values
df[confirmed_numeric].isnull().sum()


# # Impute missing values

# In[38]:


# Print out columns with null values
for col, null_values in zip(df.columns, df.isnull().sum()):
    if null_values:
        print(col, null_values)


# In[ ]:





# In[39]:


# Non-scaled imputing
n_neighbors = 5
imputer = KNNImputer(n_neighbors=n_neighbors, weights='uniform')
X = imputer.fit_transform(df)
for col, values in zip(df.columns, X.T): df[col] = values


# In[40]:


# Print out columns with null values
for col, null_values in zip(df.columns, df.isnull().sum()):
    if null_values:
        print(col, null_values)


# In[41]:


# Correct the binary features
df['R classification'] = np.where(df['R classification'] > 0.5, 1, 0)
df['Histology description_Adenocarcinoma'] = np.where(df['Histology description_Adenocarcinoma'] > 0.5, 1, 0)
df['Mucinous'] = np.where(df['Mucinous'] > 0.5, 1, 0)


# # Create copy without blood samples

# In[42]:


blood_test


# In[43]:


# df.columns[-1] = 'mrT (TNM ed.7)_3-4'
df.rename(columns={
    'mrT    (TNM ed.7)_3-4': 'mrT (TNM ed.7)_3-4', 
    'p/ypT       (TNM ed.7)_3-4-4a': 'p/ypT (TNM ed.7)_3-4-4a'
}, inplace=True)


# In[44]:


# Also a version without the blood test data
to_drop_blood = [
    'CEA baseline', 'CRP baseline', 'Hemoglobin (g/dl)', 'Thrombocytes (10ˆ9/L)', 
    'Leukocytes (10ˆ9/L)', 'Albumin (g/L)', 'Sodium (mmol/L)', 'Potassium (mmol/L)', 
    'Creatinine (umol/L)', 'Bilirubin (umol/L)', 'ALT (U/L)', 'GT (U/L)', 'ALP (U/L)',
    'Blood type_a_plus', 'Blood type_b_plus', 'Blood type_0_plus', 'Blood type_other'
]
df_no_blood = df.drop(to_drop_blood, axis=1)


# # Now let's clean the column names so they work in R.

# In[45]:


### df.columns = formatting_tools.clean_col_names(df.columns)
#df_no_blood.columns = formatting_tools.clean_col_names(df_no_blood.columns)

df_no_blood.columns


# # Now save the cleaned data set

# In[46]:


df.to_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/oxytarget/oxytarget.csv')
df_no_blood.to_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/oxytarget/oxytarget_no_blood_samples.csv')
targets.to_csv('/Users/mikkorekstad/Skole/master_data/prepared_data/oxytarget/oxytarget_response.csv')





