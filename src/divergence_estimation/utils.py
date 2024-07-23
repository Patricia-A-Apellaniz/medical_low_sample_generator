import os

import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import norm


def store_results(results_folder, cfg=None):
    kl_folder = results_folder + '/kl'
    js_folder = results_folder + '/js'

    # list of elements in the folder
    param_list = os.listdir(kl_folder)

    df_kl = pd.DataFrame()
    df_js = pd.DataFrame()
    df_kl_std = pd.DataFrame()
    df_kl_ci = pd.DataFrame()
    df_js_std = pd.DataFrame()
    df_js_ci = pd.DataFrame()

    for param in tqdm(param_list):
        # List of elements in the folder
        kl_seeds = os.listdir(kl_folder + '/' + param)
        df_seed_kl = pd.DataFrame()
        df_seed_js = pd.DataFrame()
        # number of elements in kl seeds that contains .csv
        num_seeds = len([seed for seed in kl_seeds if '.csv' in seed])
        feat_dict = {}
        if num_seeds > 0:
            for seed in kl_seeds:
                if '.csv' in seed:
                    # merge dataframe
                    # read csv with header
                    df_i = pd.read_csv(kl_folder + '/' + param + '/' + seed)
                    df_j = pd.read_csv(js_folder + '/' + param + '/' + seed)
                    df_i['val_seed'] = seed.split('_')[1]
                    df_j['val_seed'] = seed.split('_')[1]
                    df_i['name'] = '_'.join(seed.split('_')[2:]).split('.csv')[0]
                    df_j['name'] = '_'.join(seed.split('_')[2:]).split('.csv')[0]
                    # concatenate dataframe df_i to df_seed
                    df_seed_kl = pd.concat([df_seed_kl, df_i])
                    df_seed_js = pd.concat([df_seed_js, df_j])
                    if cfg.print_feat_js:
                        feat = seed
                        js = df_j['JS Discriminator'][0]
                        feat_dict[feat] = js
                        print(f'JS for {feat}: {js}')

            # Sort dictionary by value
            feat_dict = {k: v for k, v in sorted(feat_dict.items(), key=lambda item: item[1])}
            print(feat_dict)
            # Promediate over seeds
            df_aux = df_seed_kl.copy()
            df_aux_js = df_seed_js.copy()

            if 'val_seed' in df_aux.columns:
                # # Keep best seed
                df_seed_kl = pd.DataFrame(df_aux.groupby(['n', 'm', 'l', 'name']).mean()).reset_index()
                df_seed_kl_std = pd.DataFrame(df_aux.groupby(['n', 'm', 'l', 'name']).std()).reset_index()
                df_seed_js = pd.DataFrame(df_aux_js.groupby(['n', 'm', 'l', 'name']).mean()).reset_index()
                df_seed_js_std = pd.DataFrame(df_aux_js.groupby(['n', 'm', 'l', 'name']).std()).reset_index()
            else:
                if len(df_seed_kl) < 1:
                    print('Empty dataframe')
                df_seed_kl = df_seed_kl.groupby(['n', 'm', 'l']).mean()
                df_seed_kl_std = df_aux.groupby(['n', 'm', 'l']).std()
                df_aux_js = df_seed_js.copy()
                df_seed_js = df_seed_js.groupby(['n', 'm', 'l']).mean()
                df_seed_js_std = df_aux_js.groupby(['n', 'm', 'l']).std()

            num_seeds = len(kl_seeds)
            z_score = norm.ppf(1 - 0.05 / 2)
            df_seed_kl_ci = df_seed_kl_std.copy()
            df_seed_kl_ci['KL Discriminator'] = z_score * (df_seed_kl_std['KL Discriminator'] / np.sqrt(num_seeds))

            df_kl = pd.concat([df_kl, df_seed_kl])
            if len(df_kl) < 1:
                print('Empty dataframe')
            df_kl_std = pd.concat([df_kl_std, df_seed_kl_std])
            df_kl_ci = pd.concat([df_kl_ci, df_seed_kl_ci])

            # Plot the results using confidence intervals
            df_seed_js_ci = df_seed_js_std.copy()
            df_seed_js_ci['JS Discriminator'] = z_score * (df_seed_js_std['JS Discriminator'] / np.sqrt(num_seeds))

            df_js = pd.concat([df_js, df_seed_js])
            df_js_std = pd.concat([df_js_std, df_seed_js_std])
            df_js_ci = pd.concat([df_js_ci, df_seed_js_ci])

        else:
            print(f'No results for this configuration {param}')
    # Sort df by index
    df_kl = df_kl.sort_index()
    df_kl_std = df_kl_std.sort_index()
    df_kl_ci = df_kl_ci.sort_index()
    # df_js = df_js.sort_index()
    # Save to csv
    df_kl.to_csv(results_folder + '/kl.csv', index=False)
    df_kl_std.to_csv(results_folder + '/kl_std.csv', index=False)
    df_kl_ci.to_csv(results_folder + '/kl_ci.csv', index=False)

    # Sort df by index
    df_js = df_js.sort_index()
    df_js_std = df_js_std.sort_index()
    df_js_ci = df_js_ci.sort_index()
    # Save to csv
    df_js.to_csv(results_folder + '/js.csv', index=False)
    df_js_std.to_csv(results_folder + '/js_std.csv', index=False)
    df_js_ci.to_csv(results_folder + '/js_ci.csv', index=False)
