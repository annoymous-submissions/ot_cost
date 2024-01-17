#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def splitData(frac_pos, frac_neg):
    
    ##load features and cohort data
    path = ''
    data = pd.read_csv(path+'dataset.csv')
    cohort = pd.read_csv(path+'cohort_use.csv')
    ##create full dataset then split features and outcomes (needed as not full cohort in the dataset) 
    full_df = data.merge(cohort[['hadm_id', 'outcome']], on = 'hadm_id').set_index('hadm_id')
    full_df.drop('hours_since_admit', axis = 1, inplace = True)

    df_1 = pd.concat([full_df.groupby('outcome').get_group(0).sample(frac = frac_neg, random_state=1),
             full_df.groupby('outcome').get_group(1).sample(frac = frac_pos, random_state=1)])
    df_1_1, df_1_2 = random_split(df_1)
    df_2 = full_df.loc[~full_df.index.isin(df_1.index)]
    df_2_1, df_2_2 = random_split(df_2)
    return df_1_1, df_1_2, df_2_1, df_2_2


# In[4]:


def random_split(df):
    df_1 = df.sample(frac = 0.5)
    df_2 = df.drop(df_1.index)
    return df_1, df_2


# In[5]:


def feature_label_split(df):
    X = df.loc[:, ~df.columns.isin(['outcome', 'hours_since_admit'])]
    y = df['outcome']
    return X, y



def splitDataCredit(data, frac_pos, frac_neg, federated = False):
    
    df_1 = pd.concat([data.groupby('Class').get_group(0).sample(frac = frac_neg, random_state=2),
             data.groupby('Class').get_group(1).sample(frac = frac_pos, random_state=2)])
    df_2 = data.loc[~data.index.isin(df_1.index)]
    if federated:
        df_1_1, df_1_2 = random_split(df_1)
        df_2_1, df_2_2 = random_split(df_2)
        return df_1_1, df_1_2, df_2_1, df_2_2

    else:
        return df_1.sample(frac = 1), df_2.sample(frac = 1)


# In[4]:


def random_split(df):
    df_1 = df.sample(frac = 0.5)
    df_2 = df.drop(df_1.index)
    return df_1, df_2





