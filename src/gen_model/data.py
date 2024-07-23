# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 06/09/2023


# Packages to import
import math

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# -----------------------------------------------------------
#                   DATA NORMALIZATION
# -----------------------------------------------------------

def normalize_data(raw_df, feat_distributions, df_gen=None):
    num_patient, num_feature = raw_df.shape
    norm_df = raw_df.copy()
    if df_gen is not None:
        norm_gen = df_gen.copy()
    for i in range(num_feature):
        values = raw_df.iloc[:, i]
        if len(values) == 1 and math.isnan(values[0]):
            values = np.zeros((1,))
        no_nan_values = values[~pd.isnull(values)].values
        if feat_distributions[i][0] == 'gaussian':
            loc = np.mean(no_nan_values)
            scale = np.std(no_nan_values)
        elif feat_distributions[i][0] == 'bernoulli':
            if len(np.unique(no_nan_values)) == 1:
                continue
            loc = np.amin(no_nan_values)
            scale = np.amax(no_nan_values) - np.amin(no_nan_values)
        elif feat_distributions[i][0] == 'categorical':
            loc = np.amin(no_nan_values)
            scale = 1  # Do not scale
        elif feat_distributions[i][0] == 'weibull' or feat_distributions[i][0] == 'exponential':
            loc = -1 if 0 in no_nan_values else 0
            scale = 1
        elif feat_distributions[i][0] == 'log-normal':
            if raw_df.iloc[:, i].min() == 0:
                loc = 0
            else:
                loc = -1
            scale = 1
        else:
            print('Distribution ', feat_distributions[i][0], ' not normalized')
            param = np.array([0, 1])  # loc = 0, scale = 1, means that data is not modified!!
            loc = param[-2]
            scale = param[-1]
        if feat_distributions[i][0] == 'weibull' or feat_distributions[i][0] == 'exponential':
            if 0 in raw_df.iloc[:, i].values:
                raw_df.iloc[:, i] = raw_df.iloc[:, i] + 1
            else:
                # Mover datos para que empiecen en 1
                raw_df.iloc[:, i] = raw_df.iloc[:, i] - np.amin(raw_df.iloc[:, i]) + 1
            norm_df.iloc[:, i] = (raw_df.iloc[:, i]) / np.mean(raw_df.iloc[:, i])
        else:
            norm_df.iloc[:, i] = (raw_df.iloc[:, i] - loc) / scale if scale != 0 else raw_df.iloc[:, i] - loc
            if df_gen is not None:
                norm_gen.iloc[:, i] = (norm_gen.iloc[:, i] - loc) / scale if scale != 0 else norm_gen.iloc[:, i] - loc

    if df_gen is not None:
        return norm_df.reset_index(drop=True), norm_gen.reset_index(drop=True)

    return norm_df.reset_index(drop=True)


def denormalize_data(raw_df, norm_df, feat_distributions):
    num_feature = raw_df.shape[1]
    denorm_df = norm_df.copy()
    for i in range(num_feature):
        values = raw_df.iloc[:, i]
        no_nan_values = values[~np.isnan(values)].values
        if feat_distributions[i][0] == 'gaussian':
            loc = np.mean(no_nan_values)
            scale = np.std(no_nan_values)
        elif feat_distributions[i][0] == 'bernoulli':
            loc = np.amin(raw_df.iloc[:, i])
            scale = np.amax(no_nan_values) - np.amin(no_nan_values)
        elif feat_distributions[i][0] == 'categorical':
            loc = np.amin(no_nan_values)
            scale = 1  # Do not scale
        elif feat_distributions[i][0] == 'weibull' or feat_distributions[i][0] == 'exponential':
            loc = 1 if 0 in no_nan_values else 0
            scale = 1
        elif feat_distributions[i][0] == 'log-normal':
            loc = 1
            scale = 1
        else:
            print('Distribution ', feat_distributions[i][0], ' not normalized')
            param = np.array([0, 1])  # loc = 0, scale = 1, means that data is not modified!!
            loc = param[-2]
            scale = param[-1]
        if feat_distributions[i][0] == 'weibull' or feat_distributions[i][0] == 'exponential':
            # Multiplicar datos normalizados por la media de los datos en crudo
            denorm_df.iloc[:, i] = norm_df.iloc[:, i] * np.mean(no_nan_values)
            if 0 in no_nan_values:
                denorm_df.iloc[:, i] = denorm_df.iloc[:, i] - 1
            else:
                # Mover datos para que empiecen en 1
                denorm_df.iloc[:, i] = denorm_df.iloc[:, i] + np.amin(no_nan_values) - 1
            norm_df.iloc[:, i] = (raw_df.iloc[:, i]) / np.mean(raw_df.iloc[:, i])
        else:
            if scale != 0:
                denorm_df.iloc[:, i] = norm_df.iloc[:, i] * scale + loc
            else:
                denorm_df.iloc[:, i] = norm_df.iloc[:, i] + loc
    return denorm_df


# -----------------------------------------------------------
#                   DATA SPLITTING
# -----------------------------------------------------------
def split_data(data, mask):
    train_data, val_data, train_mask, val_mask = train_test_split(data, mask, test_size=0.2, random_state=0)
    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    train_mask.reset_index(drop=True, inplace=True)
    val_mask.reset_index(drop=True, inplace=True)
    return train_data, train_mask, val_data, val_mask


# -----------------------------------------------------------
#                   Sampling data
# -----------------------------------------------------------
def sample_group(group):
    return group.sample(1, random_state=1)


def sample_cat(df, cols, n):
    if n > df.shape[0]:
        print('There are not enough samples to sample ', n, ' samples. Returning all samples available.')
        return df
    elif n == 0 or n == len(df):  # Note: the last condition means that if n is the actual number of samples, do nothing
        return df
    sampled_df = pd.DataFrame()
    for feat in cols:
        sampled_dfi = (df.groupby([feat]).apply(sample_group))
        sampled_dfi = sampled_dfi.reset_index(drop=True)
        sampled_df = pd.concat([sampled_df, sampled_dfi])
    # samples that are in df but not in sampled_df
    df_dif = pd.merge(df, sampled_df, how='outer', indicator=True).query('_merge=="left_only"').drop('_merge', axis=1)
    if n - len(sampled_df) > 0 and n - len(sampled_df) <= len(df_dif):
        df_dif_sample = df_dif.sample(n=(n - len(sampled_df)), random_state=1)
        df_sample = pd.concat([sampled_df, df_dif_sample])
    elif n - len(sampled_df) > len(df_dif):
        print('WARNING: sampling must take repeated samples to achieve minimum number of samples required')
        df_dif_sample = df_dif.sample(n=(n - len(sampled_df)), random_state=1, replace=True)
        df_sample = pd.concat([sampled_df, df_dif_sample])
    else:
        df_sample = sampled_df.sample(n=n, random_state=1).reset_index(drop=True)
        print('WARNING: not enough samples to sample ', n, ' samples with all categories, minimum would be ',
              len(sampled_df), '. Returning ', n, ' samples.')
    return df_sample
