# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 05/03/2024


# Packages to import
import os
import pickle

import numpy as np
import pandas as pd

from pycox import datasets
from data_manager import DataManager


def preprocess_metabric(dataset_name):
    # Load data
    raw_df = datasets.metabric.read_df()
    raw_df = raw_df.sample(frac=1).reset_index(drop=True)  # Reorder the data randomly in case patients are ordered

    # Transform covariates and create df
    label = raw_df[['event']]
    time = raw_df[['duration']]
    raw_df = raw_df.drop(labels=['event', 'duration'], axis=1)
    survival_df = time.join(label)
    raw_df = raw_df.join(survival_df)
    raw_df = raw_df.rename(columns={raw_df.columns[-1]: 'event', raw_df.columns[-2]: 'time'})

    # Transform covariates and create df
    df = raw_df.copy()

    # Create data manager object
    data_manager = DataManager(dataset_name, raw_df, df)

    # Obtain feature distributions
    feat_distributions = []
    for i in range(df.shape[1]):
        values = df.iloc[:, i].unique()
        no_nan_values = values[~pd.isnull(values)]
        if no_nan_values.size <= 2 and np.all(np.sort(no_nan_values).astype(int) ==
                                              np.array(range(no_nan_values.min().astype(int),
                                                             no_nan_values.min().astype(int) + len(no_nan_values)))):
            feat_distributions.append(('bernoulli', 1))
        elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
            # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
            if no_nan_values.dtype == 'float64':
                no_nan_values = no_nan_values.astype(int)
            if np.unique(no_nan_values).size < 50 and np.amin(no_nan_values) == 0:
                feat_distributions.append(('categorical', (np.max(no_nan_values) + 1).astype(int)))
            else:
                feat_distributions.append(('gaussian', 2))
        else:
            feat_distributions.append(('gaussian', 2))
    data_manager.set_feat_distributions(feat_distributions)

    # Normalize, impute data
    # Necessary to impute before normalization because of the categorical variables treated as gaussian.
    data_manager.norm_df = data_manager.transform_data(df)
    data_manager.imp_norm_df = data_manager.impute_data(data_manager.norm_df)

    # Create metadata for ctgan and tvae
    data_manager.get_metadata()

    return data_manager


def preprocess_nwtco(dataset_name):
    # Load data
    raw_df = datasets.nwtco.read_df(
        processed=False)  # Processed is set to False because there is a bug. The following lines are copied from pycox library
    raw_df = raw_df.sample(frac=1).reset_index(drop=True)  # Reorder the data randomly in case patients are ordered

    # Transform covariates and create df
    raw_df = (raw_df.assign(instit_2=raw_df['instit'] - 1, histol_2=raw_df['histol'] - 1, study_4=raw_df['study'] - 3,
                            stage=raw_df['stage'].astype('category')).drop(
        ['rownames', 'seqno', 'instit', 'histol', 'study'], axis=1))
    raw_df['stage'] = raw_df['stage'].astype('int')
    label = raw_df[['rel']]
    time = raw_df[['edrel']]
    raw_df = raw_df.drop(labels=['rel', 'edrel'], axis=1)
    survival_df = time.join(label)
    raw_df = raw_df.join(survival_df)
    raw_df = raw_df.rename(columns={raw_df.columns[-1]: 'event', raw_df.columns[-2]: 'time'})

    # Transform covariates and create df
    df = raw_df.copy()
    mapping_info = {}
    df['in.subcohort'], classes = df['in.subcohort'].factorize()
    mapping_info['in.subcohort'] = np.array(classes.values)

    # Create data manager object
    data_manager = DataManager(dataset_name, raw_df, df, mapping_info=mapping_info)

    # Obtain feature distributions
    feat_distributions = []
    for i in range(df.shape[1]):
        values = df.iloc[:, i].unique()
        no_nan_values = values[~pd.isnull(values)]
        if no_nan_values.size <= 2 and np.all(np.sort(no_nan_values).astype(int) ==
                                              np.array(range(no_nan_values.min().astype(int),
                                                             no_nan_values.min().astype(int) + len(no_nan_values)))):
            feat_distributions.append(('bernoulli', 1))
        elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
            # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
            if no_nan_values.dtype == 'float64':
                no_nan_values = no_nan_values.astype(int)
            if np.unique(no_nan_values).size < 50 and np.amin(no_nan_values) == 0:
                feat_distributions.append(('categorical', (np.max(no_nan_values) + 1).astype(int)))
            else:
                feat_distributions.append(('gaussian', 2))
        else:
            feat_distributions.append(('gaussian', 2))
    data_manager.set_feat_distributions(feat_distributions)

    # Normalize, impute data
    # Necessary to impute before normalization because of the categorical variables treated as gaussian.
    data_manager.norm_df = data_manager.transform_data(df)
    data_manager.imp_norm_df = data_manager.impute_data(data_manager.norm_df)

    # Create metadata for ctgan and tvae
    data_manager.get_metadata()

    return data_manager


def preprocess_std(dataset_name, args):
    # Load data
    data_filename = args['input_dir'] + 'sa_data/std/std.csv'
    raw_df = pd.read_csv(data_filename, sep=',', index_col=0)
    raw_df = raw_df.sample(frac=1).reset_index(drop=True)  # Reorder the data randomly in case patients are ordered

    # Transform covariates and create df
    label = raw_df[['rinfct']]
    time = raw_df[['time']]
    raw_df = raw_df.drop(labels=['obs', 'rinfct', 'time'], axis=1)
    survival_df = time.join(label)
    raw_df = raw_df.join(survival_df)
    raw_df = raw_df.rename(columns={raw_df.columns[-1]: 'event', raw_df.columns[-2]: 'time'})

    # Transform covariates and create df
    df = raw_df.copy()
    mapping_info = {}
    df['race'], classes = df['race'].factorize()
    mapping_info['race'] = np.array(classes.values)
    df['race'] = df['race'].replace(-1, np.nan)
    df['marital'], classes = df['marital'].factorize()
    mapping_info['marital'] = np.array(classes.values)
    df['marital'] = df['marital'].replace(-1, np.nan)
    df['iinfct'], classes = df['iinfct'].factorize()
    mapping_info['iinfct'] = np.array(classes.values)
    df['iinfct'] = df['iinfct'].replace(-1, np.nan)
    df['condom'], classes = df['condom'].factorize()
    mapping_info['condom'] = np.array(classes.values)
    df['condom'] = df['condom'].replace(-1, np.nan)

    # Create data manager object
    data_manager = DataManager(dataset_name, raw_df, df, mapping_info=mapping_info)

    # Obtain feature distributions
    feat_distributions = []
    for i in range(df.shape[1]):
        values = df.iloc[:, i].unique()
        no_nan_values = values[~pd.isnull(values)]
        if no_nan_values.size <= 2 and np.all(np.sort(no_nan_values).astype(int) ==
                                              np.array(range(no_nan_values.min().astype(int),
                                                             no_nan_values.min().astype(int) + len(no_nan_values)))):
            feat_distributions.append(('bernoulli', 1))
        elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
            # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
            if no_nan_values.dtype == 'float64':
                no_nan_values = no_nan_values.astype(int)
            if np.unique(no_nan_values).size < 50 and np.amin(no_nan_values) == 0:
                feat_distributions.append(('categorical', (np.max(no_nan_values) + 1).astype(int)))
            else:
                feat_distributions.append(('gaussian', 2))
        else:
            feat_distributions.append(('gaussian', 2))
    data_manager.set_feat_distributions(feat_distributions)

    # Normalize, impute data
    # Necessary to impute before normalization because of the categorical variables treated as gaussian.
    data_manager.norm_df = data_manager.transform_data(df)
    data_manager.imp_norm_df = data_manager.impute_data(data_manager.norm_df)

    # Create metadata for ctgan and tvae
    data_manager.get_metadata()

    return data_manager


def preprocess_pbc(dataset_name, args):
    # Load data
    data_filename = args['input_dir'] + 'sa_data/pbc/pbc.csv'
    raw_df = pd.read_csv(data_filename, sep=',', index_col=0)
    raw_df = raw_df.sample(frac=1).reset_index(drop=True)  # Reorder the data randomly in case patients are ordered

    # Transform covariates and create df
    raw_df['edema'] = raw_df['edema'].apply(
        lambda x: 0 if x < 0.25 else (2 if x > 0.75 else 1))  # Make this a categorical variable (3 values)
    label = raw_df[['status']]
    time = raw_df[['days']] / 30
    raw_df = raw_df.drop(labels=['status', 'days'], axis=1)
    survival_df = time.join(label)
    raw_df = raw_df.join(survival_df)
    raw_df = raw_df.rename(columns={raw_df.columns[-1]: 'event', raw_df.columns[-2]: 'time'})

    # Transform covariates and create df
    df = raw_df.copy()
    mapping_info = {}
    df['treatment'], classes = df['treatment'].factorize()
    mapping_info['treatment'] = np.array(classes.values)
    df['treatment'] = df['treatment'].replace(-1, np.nan)
    df['stage'], classes = df['stage'].factorize()
    mapping_info['stage'] = np.array(classes.values)
    df['stage'] = df['stage'].replace(-1, np.nan)

    # Create data manager object
    data_manager = DataManager(dataset_name, raw_df, df, mapping_info=mapping_info)

    # Obtain feature distributions
    feat_distributions = []
    for i in range(df.shape[1]):
        values = df.iloc[:, i].unique()
        no_nan_values = values[~pd.isnull(values)]
        if no_nan_values.size <= 2 and np.all(np.sort(no_nan_values).astype(int) ==
                                              np.array(range(no_nan_values.min().astype(int),
                                                             no_nan_values.min().astype(int) + len(no_nan_values)))):
            feat_distributions.append(('bernoulli', 1))
        elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
            # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
            if no_nan_values.dtype == 'float64':
                no_nan_values = no_nan_values.astype(int)
            if np.unique(no_nan_values).size < 50 and np.amin(no_nan_values) == 0:
                feat_distributions.append(('categorical', (np.max(no_nan_values) + 1).astype(int)))
            else:
                feat_distributions.append(('gaussian', 2))
        else:
            feat_distributions.append(('gaussian', 2))
    data_manager.set_feat_distributions(feat_distributions)

    # Normalize, impute data
    # Necessary to impute before normalization because of the categorical variables treated as gaussian.
    data_manager.norm_df = data_manager.transform_data(df)
    data_manager.imp_norm_df = data_manager.impute_data(data_manager.norm_df)

    # Create metadata for ctgan and tvae
    data_manager.get_metadata()

    return data_manager


def preprocess_gbsg(dataset_name):
    # Load data
    raw_df = datasets.gbsg.read_df()
    raw_df = raw_df.sample(frac=1).reset_index(drop=True)  # Reorder the data randomly in case patients are ordered

    # Transform covariates and create df
    label = raw_df[['event']]
    time = raw_df[['duration']]
    raw_df = raw_df.drop(labels=['event', 'duration'], axis=1)
    survival_df = time.join(label)
    raw_df = raw_df.join(survival_df)
    raw_df = raw_df.rename(columns={raw_df.columns[-1]: 'event', raw_df.columns[-2]: 'time'})

    # Transform covariates and create df
    df = raw_df.copy()

    # Create data manager object
    data_manager = DataManager(dataset_name, raw_df, df)

    # Obtain feature distributions
    feat_distributions = []
    for i in range(df.shape[1]):
        values = df.iloc[:, i].unique()
        no_nan_values = values[~pd.isnull(values)]
        if no_nan_values.size <= 2 and np.all(np.sort(no_nan_values).astype(int) ==
                                              np.array(range(no_nan_values.min().astype(int),
                                                             no_nan_values.min().astype(int) + len(no_nan_values)))):
            feat_distributions.append(('bernoulli', 1))
        elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
            # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
            if no_nan_values.dtype == 'float64':
                no_nan_values = no_nan_values.astype(int)
            if np.unique(no_nan_values).size < 50 and np.amin(no_nan_values) == 0:
                feat_distributions.append(('categorical', (np.max(no_nan_values) + 1).astype(int)))
            else:
                feat_distributions.append(('gaussian', 2))
        else:
            feat_distributions.append(('gaussian', 2))
    data_manager.set_feat_distributions(feat_distributions)

    # Normalize, impute data
    # Necessary to impute before normalization because of the categorical variables treated as gaussian.
    data_manager.norm_df = data_manager.transform_data(df)
    data_manager.imp_norm_df = data_manager.impute_data(data_manager.norm_df)

    # Create metadata for ctgan and tvae
    data_manager.get_metadata()

    return data_manager


def preprocess_whas(dataset_name, args):
    # Load data
    data_filename = args['input_dir'] + 'sa_data/whas/whas.csv'
    raw_df = pd.read_csv(data_filename, sep=',')
    raw_df = raw_df.sample(frac=1).reset_index(drop=True)  # Reorder the data randomly in case patients are ordered

    # Transform covariates and create df
    df = raw_df.copy()

    # Create data manager object
    data_manager = DataManager(dataset_name, raw_df, df)

    # Obtain feature distributions
    feat_distributions = []
    for i in range(df.shape[1]):
        values = df.iloc[:, i].unique()
        no_nan_values = values[~pd.isnull(values)]
        if no_nan_values.size <= 2 and np.all(np.sort(no_nan_values).astype(int) ==
                                              np.array(range(no_nan_values.min().astype(int),
                                                             no_nan_values.min().astype(int) + len(no_nan_values)))):
            feat_distributions.append(('bernoulli', 1))
        elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
            # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
            if no_nan_values.dtype == 'float64':
                no_nan_values = no_nan_values.astype(int)
            if np.unique(no_nan_values).size < 50 and np.amin(no_nan_values) == 0:
                feat_distributions.append(('categorical', (np.max(no_nan_values) + 1).astype(int)))
            else:
                feat_distributions.append(('gaussian', 2))
        else:
            feat_distributions.append(('gaussian', 2))
    data_manager.set_feat_distributions(feat_distributions)

    # Normalize, impute data
    # Necessary to impute before normalization because of the categorical variables treated as gaussian.
    data_manager.norm_df = data_manager.transform_data(df)
    data_manager.imp_norm_df = data_manager.impute_data(data_manager.norm_df)

    # Create metadata for ctgan and tvae
    data_manager.get_metadata()

    return data_manager


def preprocess_by_name(dataset_name):
    args = {'input_dir': './raw_data/'}
    if dataset_name == 'metabric':
        return preprocess_metabric(dataset_name)
    elif dataset_name == 'nwtco':
        return preprocess_nwtco(dataset_name)
    elif dataset_name == 'gbsg':
        return preprocess_gbsg(dataset_name)
    elif dataset_name == 'whas':
        return preprocess_whas(dataset_name, args)
    elif dataset_name == 'pbc':
        return preprocess_pbc(dataset_name, args)
    elif dataset_name == 'std':
        return preprocess_std(dataset_name, args)
    else:
        raise ValueError('Dataset not found')


if __name__ == '__main__':
    # Preprocess and save each dataset in the corresponding folder; run this code to obtain the preprocessed data
    for dataset_name in ['metabric', 'gbsg', 'whas', 'nwtco', 'pbc', 'std']:
        data_manager = preprocess_by_name(dataset_name)
        save_path = './processed_data/sa_data/' + str(dataset_name) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save dictionary with metadata needed later
        with open(save_path + 'metadata.pkl', 'wb') as f:
            pickle.dump({'feat_distributions': data_manager.feat_distributions, 'mask': data_manager.mask, 'metadata': data_manager.metadata}, f)

        # Save data
        data_manager.imp_norm_df.to_csv(save_path + 'preprocessed_data_all.csv', index=False)
        print(f'{dataset_name} saved')