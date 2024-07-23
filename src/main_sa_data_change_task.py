import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from copy import deepcopy
from tabulate import tabulate
from colorama import Fore, Style
from scipy.stats import ttest_ind_from_stats
from utility_validation import evaluate_classification_metrics


if __name__ == '__main__':
    ## ILLUSTRATION ON HOW THE JS DIVERGENCE REFLECTS POSSIBLE CHANGES IN UTILITY
    # Here, I compare the classification performance of the real data vs the low_data synthetic data, to assess the differences in utility across different tasks in the same dataset

    datasets = ['metabric', 'gbsg', 'whas', 'nwtco', 'pbc', 'std']
    args = {'output_dir': os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'results/sa_data_change_task'),
            'input_dir': os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'data/processed_data/sa_data'),
            'n_splits': 5,  # Number of splits for the cross-validation
            }

    train = not True
    show_results = True

    for dataset in datasets:
        train_data = pd.read_csv(os.path.join(args['input_dir'], dataset, 'preprocessed_data_all.csv'))
        # Use always a 20% of the data for validation
        real_val_df = train_data.sample(frac=0.2, random_state=0)  # Data used for validation, always the same size, for comparison among all methods

        # Load synthetic data without methodology
        best_seed = pd.read_csv(os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'results/sa_data', 'low_data', dataset, 'vae', 'best_parameters.csv'))['seed'].iloc[0]
        syn_df_low = pd.read_csv(os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'results/sa_data', 'low_data', dataset, 'vae', 'seed_' + str(best_seed) + '_gen_data.csv'))

        # Load synthetic data with methodology
        best_seed = pd.read_csv(os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'results/sa_data', 'drs', dataset, 'vae', 'best_parameters.csv'))['seed'].iloc[0]
        syn_df_high = pd.read_csv(os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'results/sa_data', 'drs', dataset, 'vae', 'seed_' + str(best_seed) + '_gen_data.csv'))

        cat_cols = []
        for name in syn_df_low.columns:
            if len(syn_df_low[name].unique()) > 1:
                if syn_df_low[name].apply(float.is_integer).all() and len(syn_df_low[name].unique()) < 10:
                    cat_cols.append(name)

        # Remove the time variable if it is discrete
        if 'time' in cat_cols:
            cat_cols.remove('time')

        if train:
            for idx_to_predict in range(len(cat_cols)):  # Evaluate the prediction problem for each categorical variable
                print(f"Predicting variable {cat_cols[idx_to_predict]} of dataset {dataset}, variables remaining: {len(cat_cols) - idx_to_predict - 1}")
                args_val = deepcopy(args)
                target_var = cat_cols[idx_to_predict] # The variable name to be predicted
                args_val['output_dir'] = os.path.join(args['output_dir'], dataset, target_var)
                args_val['dataset_name'] = dataset
                # Now, reorder the variables: move to last position the target variable
                cols = real_val_df.columns.tolist()
                cols.remove(target_var)
                cols.append(target_var)
                args_val['real_df'] = syn_df_low[cols]  # NOTE: the name is "real", but it is the synthetic data in the low case
                args_val['util_val_df'] = real_val_df[cols]
                args_val['syn_df'] = syn_df_high[cols]  # NOTE: the name is "syn", but it is the synthetic data in the high case
                evaluate_classification_metrics(args_val)

        if show_results:
            print('\n\nResults for dataset ' + Fore.BLUE + dataset + Style.RESET_ALL)
            ucases = ['real', 'synth']  # Note: the real case is the low_data synthetic data, the synth case is the high_data synthetic data
            diff_synth = []
            table = []
            for idx_to_predict in range(len(cat_cols)):
                target_var = cat_cols[idx_to_predict]
                output_dir = os.path.join(args['output_dir'], dataset, target_var)
                df = pd.read_csv(os.path.join(output_dir, 'classification_metrics.csv'))
                acc = [df.loc[df['case'] == ucase]['acc'].values[0] for ucase in ucases]
                acc_std = [df.loc[df['case'] == ucase]['acc_std'].values[0] for ucase in ucases]

                # Test the acc between the real and synth cases
                test_synth = ttest_ind_from_stats(acc[0], acc_std[0], args['n_splits'], acc[1], acc_std[1],
                                                  args['n_splits'], equal_var=False, alternative='two-sided')  # Bilateral test: to test if the methodology is better or worse than the low_data case

                if test_synth.pvalue < 0.01:
                    if acc[1] > acc[0]:
                        print(
                            f"Significant ADVANTAGE for methodology for variable {target_var}: low: {acc[0]} ({acc_std[0]}) vs meth:{acc[1]} ({acc_std[1]})")
                    else:
                        print(
                            f"Significant DISADVANTAGE for methodology for variable {target_var}: low: {acc[0]} ({acc_std[0]}) vs meth:{acc[1]} ({acc_std[1]})")
                else:
                    print(
                        f"No significant difference between low_data and methodology cases for variable {target_var}: {acc[0]} ({acc_std[0]}) vs {acc[1]} ({acc_std[1]})")

                diff_synth.append(acc[1] - acc[0])
                t = [target_var] + [f"{acc[0]:.3f} ({acc_std[0]:.3f})"] + [f"{acc[1]:.3f} ({acc_std[1]:.3f})"]
                # t = [f"{acc[0]:.3f} ({acc_std[0]:.3f})"] + [f"{acc[1]:.3f} ({acc_std[1]:.3f})"]
                table.append(t)

            plt.hist(diff_synth, bins=10, alpha=0.5, label='diff')
            plt.legend(loc='best')
            plt.savefig(os.path.join(args['output_dir'], dataset, 'diff_synth.png'))

            print(tabulate(table, headers=['Big data', 'Low data'], tablefmt='orgtbl'))


