# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 05/03/2024


# Packages to import
import os
import pickle

import numpy as np
import pandas as pd

from data_manager import DataManager


def preprocess_clas(dataset_name):
    # Load data
    data_filename = './raw_data/class_data/' + dataset_name + '.csv'
    raw_df = pd.read_csv(data_filename, sep=',')
    raw_df = raw_df.sample(frac=1).reset_index(drop=True)  # Reorder the data randomly in case patients are ordered

    # Transform covariates and create df
    if dataset_name == '3_data':
        raw_df = raw_df.drop(columns=['id'])
        df = raw_df.copy()
        mapping_info = {}  # Data is already preprocessed
    elif dataset_name == '4_data':
        raw_df = raw_df.drop(columns=['encounter_id', 'patient_nbr', 'weight', 'max_glu_serum', 'A1Cresult'])
        df = raw_df.copy()
        mapping_info = {}
        for feat in ['race', 'gender', 'age', 'payer_code', 'medical_specialty', 'metformin', 'repaglinide',
                     'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                     'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
                     'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                     'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
                     'diabetesMed', 'readmitted']:
            df[feat], classes = df[feat].factorize()
            mapping_info[feat] = np.array(classes.values)

        # For diag_1, diag_2 and diag_3, we delete all patient that have a string as indicator instead of a number
        for d in ['diag_1', 'diag_2', 'diag_3']:
            df = df[~df[d].str.contains('V', na=False)]
            df = df[~df[d].str.contains('E', na=False)]
            df = df[~df[d].str.contains('\?', na=False)]
            df[d] = df[d].astype(float)
    elif dataset_name == '7_data':
        raw_df = raw_df.drop(columns=['id'])
        df = raw_df.copy()
        mapping_info = {}  # Data is already preprocessed
    else:
        raise ValueError('Dataset not found')

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
            if np.unique(no_nan_values).size < 15 and np.amin(no_nan_values) == 0:
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



if __name__ == '__main__':

    DATA_N = 10000
    DATA_M = 7500
    DATA_L = 2 * 1000
    DATA_ALL = DATA_N + DATA_M + DATA_L

    # Preprocess and save each dataset in the corresponding folder; run this code to obtain the preprocessed data
    for dataset_name in ['3_data', '4_data', '7_data']:
        data_manager = preprocess_clas(dataset_name)
        save_path = './processed_data/class_data/' + str(dataset_name) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save dictionary with metadata needed later
        with open(save_path + 'metadata.pkl', 'wb') as f:
            pickle.dump({'feat_distributions': data_manager.feat_distributions, 'mask': data_manager.mask, 'metadata': data_manager.metadata}, f)

        # Save data
        data_manager.imp_norm_df.to_csv(save_path + 'preprocessed_data_all.csv', index=False)
        data_manager.imp_norm_df[0: DATA_N].to_csv(save_path + 'preprocessed_data_n.csv', index=False)
        data_manager.imp_norm_df[DATA_N: DATA_N + DATA_M].to_csv(save_path + 'preprocessed_data_m.csv', index=False)
        data_manager.imp_norm_df[DATA_N + DATA_M: DATA_ALL].to_csv(save_path + 'preprocessed_data_l.csv', index=False)
        print(f'{dataset_name} saved')