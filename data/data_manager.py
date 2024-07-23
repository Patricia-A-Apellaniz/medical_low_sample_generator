# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 16/01/2024

# Packages to import
import numpy as np
import pandas as pd

from scipy import stats
from sklearn.svm import SVR
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sdv.metadata import SingleTableMetadata
from sklearn.linear_model import BayesianRidge


class DataManager:
    def __init__(self, dataset_name, raw_df, processed_df, mapping_info=None):
        self.dataset_name = dataset_name
        self.raw_df = raw_df
        self.processed_df = processed_df
        self.columns = self.processed_df.columns
        self.mapping_info = mapping_info
        self.imp_df = None
        self.norm_df = None
        self.imp_norm_df = None
        self.feat_distributions = None
        self.positive_gaussian_cols = None
        self.model_data = None
        self.rec_info = {}
        self.gen_info = {}
        self.generate_mask = False
        self.gen_raw_data = {}
        self.metadata = None
        self.gauss_gen_info = {}
        self.gauss_gen_raw_data = {}

        # Mask to be used during training
        if self.processed_df.isna().any().any():
            nans = self.processed_df.isna()
            self.raw_mask = nans.replace([True, False], [0, 1])

            # Check if generate_mask is True and nan values exist
            self.generate_mask = True
            self.gen_mask = {}
            self.gen_nan_raw_data = {}
            self.gauss_gen_mask = {}
            self.gauss_gen_nan_raw_data = {}
        else:
            mask = np.ones((self.processed_df.shape[0], self.processed_df.shape[1]))
            self.raw_mask = pd.DataFrame(mask, columns=self.columns)

        self.mask = self.raw_mask.copy()

    def set_feat_distributions(self, feat_distributions):
        self.feat_distributions = feat_distributions
        # Get positive gaussian columns for postprocessing purposes
        positive_columns = []
        for idx, dist in enumerate(self.feat_distributions):
            if dist[0] == 'gaussian':
                values = self.processed_df.iloc[:, idx].values.astype(float)
                non_missing_values = values[~np.isnan(values)]
                if not (non_missing_values < 0).any():
                    positive_columns.append(idx)
        self.positive_gaussian_cols = positive_columns

    # Necessary for CTGAN
    def get_metadata(self, metadata=None):
        if metadata is None:
            self.metadata = SingleTableMetadata()
            self.metadata.detect_from_dataframe(self.processed_df)
        else:
            self.metadata = metadata
        return self.metadata

    def zero_imputation(self, data):
        imp_data = data.copy()
        imp_data = imp_data.fillna(0)
        return imp_data

    def mice_imputation(self, data, model='bayesian'):
        imp_data = data.copy()
        if model == 'bayesian':
            clf = BayesianRidge()
        elif model == 'svr':
            clf = SVR()
        else:
            raise RuntimeError('MICE imputation base_model not recognized')
        imp = IterativeImputer(estimator=clf, verbose=2, max_iter=30, tol=1e-10, imputation_order='roman')
        imp_data.iloc[:, :] = imp.fit_transform(imp_data)
        return imp_data

    def statistics_imputation(self, data, norm):
        imp_data = data.copy()
        # If data comes from classification task and multitask dataset, columns size doesn't match data's columns size
        n_columns = data.columns.size if data.columns.size < self.columns.size else self.columns.size
        for i in range(n_columns):
            values = data.iloc[:, i].values
            raw_values = self.processed_df.iloc[:, i].values
            if any(pd.isnull(values)):
                no_nan_values = values[~pd.isnull(values)]
                no_nan_raw_values = raw_values[~pd.isnull(raw_values)]
                if values.dtype in [object, str] or no_nan_values.size <= 2 or np.amin(
                        np.equal(np.mod(no_nan_values, 1), 0)):
                    stats_value = stats.mode(no_nan_values, keepdims=True)[0][0]
                # If raw data has int values take mode normalized
                elif norm and np.amin(np.equal(np.mod(no_nan_raw_values, 1), 0)):
                    stats_value = stats.mode(no_nan_raw_values, keepdims=True)[0][0]
                    # Find index of stats_value in self.raw_df.iloc[:, i].values
                    idx = np.where(self.processed_df.iloc[:, i].values == stats_value)[0][0]
                    # Find which value is in idx of data.iloc[:, i].values and set this value to stats_value
                    stats_value = values[np.where(values == data.iloc[:, i].values[idx])[0][0]]
                else:
                    stats_value = no_nan_values.mean()
                imp_data.iloc[:, i] = [stats_value if pd.isnull(x) else x for x in imp_data.iloc[:, i]]

        return imp_data

    # Transform data according to raw_df
    def transform_data(self, df, denorm=False):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        transformed_df = df.copy()
        for i in range(self.processed_df.shape[1]):
            dist = self.feat_distributions[i][0]
            values = self.processed_df.iloc[:, i]
            no_nan_values = values[~pd.isnull(values)].values
            if dist == 'gaussian':
                loc = np.mean(no_nan_values)
                scale = np.std(no_nan_values)
            elif dist == 'bernoulli':
                loc = np.amin(no_nan_values)
                scale = np.amax(no_nan_values) - np.amin(no_nan_values)
            elif dist == 'categorical':
                loc = np.amin(no_nan_values)
                scale = 1  # Do not scale
            elif dist == 'weibull':
                loc = -1 if 0 in no_nan_values else 0
                scale = 0
            else:
                raise NotImplementedError('Distribution ', dist, ' not normalized!')

            if denorm:  # Denormalize
                transformed_df.iloc[:, i] = (df.iloc[:, i] * scale + loc if scale != 0
                                             else df.iloc[:, i] + loc).astype(self.processed_df.iloc[:, i].dtype)
            else:  # Normalize
                transformed_df.iloc[:, i] = (df.iloc[:, i] - loc) / scale if scale != 0 else df.iloc[:, i] - loc

        return transformed_df

    def impute_data(self, df, mode='stats', norm=True):
        # If missing data exists, impute it
        if df.isna().any().any():
            # Data imputation
            if mode == 'zero':
                imp_df = self.zero_imputation(df)
            elif mode == 'stats':
                imp_df = self.statistics_imputation(df, norm)
            else:
                imp_df = self.mice_imputation(df)
        else:
            imp_df = df.copy()

        return imp_df
