# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 12/09/2023


# Packages to import
import torch

import numpy as np
import pandas as pd

from ..data import normalize_data, denormalize_data
from sklearn.mixture import BayesianGaussianMixture
from ..base_model.vae_model import VariationalAutoencoder
from ..base_model.vae_utils import check_nan_inf, sample_from_dist


def get_positive_gaussian_columns(df, feat_distributions):
    positive_columns = []
    for idx, dist in enumerate(feat_distributions):
        if dist[0] == 'gaussian':
            values = df.iloc[:, idx].values.astype(float)
            non_missing_values = values[~np.isnan(values)]
            if not (non_missing_values < 0).any():
                positive_columns.append(idx)
    return positive_columns


def transform_features(real_df, df, feat_distributions):
    # Denormalize df
    raw_df = denormalize_data(real_df, df, feat_distributions)

    for col in range(real_df.shape[1]):
        col_name = real_df.columns[col]
        # First, transform to original dtype
        raw_df.iloc[:, col] = raw_df.iloc[:, col].astype(real_df.dtypes[col])
        # Then, round to the same number of decimals as the original data
        if real_df.dtypes[col] == 'float64':
            max_dec = 0
            for val in real_df.iloc[:, col]:
                if not np.isnan(val):
                    dec_digits = float((str(val).split('.')[1]))
                    if dec_digits == 0:
                        dec = 0
                    else:
                        dec = len(str(val).split('.')[1])
                    if dec > max_dec:
                        max_dec = dec
            raw_df.iloc[:, col] = raw_df.iloc[:, col].round(max_dec)
    return raw_df


class Generator(VariationalAutoencoder):
    """
    Module implementing Synthethic data generator
    """

    def __init__(self, params):
        # Initialize Generator parameters and modules
        super(Generator, self).__init__(params)
        self.bgm = None

    def train_latent_generator(self, x):  # Train latent generator using x data as input
        if self.bgm is not None:
            raise RuntimeWarning(['WARNING] BGM is being retrained'])

        vae_data = self.predict(x)
        mu_latent_param, log_var_latent_param = vae_data['latent_params']

        # Fit GMM to the mean
        converged = False
        bgm = None
        n_try = 0
        max_try = 100
        # NOTE: this code is for Gaussian latent space, change it if using a different one!
        while not converged and n_try < max_try:  # BGM may not converge: try different times until it converges
            # (or it reaches a max number of iterations)
            n_try += 1
            n_try += 1
            bgm = BayesianGaussianMixture(n_components=self.latent_dim, random_state=42 + n_try, reg_covar=1e-5,
                                          n_init=10, max_iter=5000).fit(mu_latent_param)  # Use only mean
            converged = bgm.converged_

        if not converged:
            print('[WARNING] BGM did not converge after ' + str(n_try + 1) + ' attempts')
            print('NOT CONVERGED')
        else:
            self.bgm = {'bgm': bgm,
                        'log_var_mean': np.mean(log_var_latent_param, axis=0)}  # BGM data to generate patients

    def generate(self, n_gen=100):
        if self.bgm is None:
            print('[WARNING] BGM  is not trained, try calling train_latent_generator before calling generate')
        else:
            mu_sample = self.bgm['bgm'].sample(n_gen)[0]
            log_var_sample = np.tile(self.bgm['log_var_mean'], (n_gen, 1))

            z = self.latent_space.sample_latent([torch.from_numpy(mu_sample).float(),
                                                 torch.from_numpy(log_var_sample).float()])
            check_nan_inf(z, 'GMM latent space')
            cov_params = self.Decoder(z)
            check_nan_inf(cov_params, 'Decoder')
            cov_params = cov_params.detach().cpu().numpy()
            cov_samples = sample_from_dist(cov_params, self.feat_distributions)
            out_data = {'z': z.detach().cpu().numpy(),
                        'cov_params': cov_params,
                        'cov_samples': cov_samples,
                        'latent_params': [mu_sample, log_var_sample]}

            return out_data

    # Remove samples that don't match the original samples' format. For example, number of decimals or negative samples
    def postprocess_data(self, real_df, gen_data, feat_distributions, bad_seed=False):
        cov_samples = gen_data['cov_samples']
        z = gen_data['z']
        cov_params = gen_data['cov_params']
        latent_params = gen_data['latent_params']
        raw_cov_samples = denormalize_data(real_df, pd.DataFrame(cov_samples), feat_distributions)
        pos_columns = get_positive_gaussian_columns(real_df, feat_distributions)

        # 1. Remove negative gaussian samples if the original data didn't have them
        if not bad_seed:
            for idx in range(cov_samples.shape[1]):
                if idx in pos_columns:
                    cov_samples = cov_samples[raw_cov_samples.iloc[:, idx] >= 0]
                    z = z[raw_cov_samples.iloc[:, idx] >= 0]
                    cov_params = cov_params[raw_cov_samples.iloc[:, idx] >= 0]
                    latent_params[0] = latent_params[0][raw_cov_samples.iloc[:, idx] >= 0]
                    latent_params[1] = latent_params[1][raw_cov_samples.iloc[:, idx] >= 0]
                    raw_cov_samples = raw_cov_samples[raw_cov_samples.iloc[:, idx] >= 0]

            if len(cov_samples) > 0:
                # 2. Round features and transform data types based on original ones
                raw_cov_samples = transform_features(real_df, pd.DataFrame(cov_samples), feat_distributions)
                # If we normalize again, transformation is lost
                cov_samples = normalize_data(pd.DataFrame(raw_cov_samples), feat_distributions)

        gen_data = {'z': z,
                    'cov_params': cov_params,
                    'cov_samples': cov_samples,
                    'raw_cov_samples': raw_cov_samples,
                    'latent_params': latent_params}
        return gen_data
