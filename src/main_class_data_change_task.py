import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from copy import deepcopy
from tabulate import tabulate
from colorama import Fore, Style
from gen_model.data import sample_cat
from scipy.stats import ttest_ind_from_stats
from utility_validation import evaluate_classification_metrics

if __name__ == '__main__':
    ## ILLUSTRATION ON HOW THE JS DIVERGENCE REFLECTS POSSIBLE CHANGES IN UTILITY
    # Here, I compare the classification performance of the low_data synthetic data vs the methodology synthetic data, to assess the differences in utility across different tasks in the same dataset

    datasets = ['3_data', '4_data', '7_data']
    args = {'output_dir': os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'results/class_data_change_task'),
            'input_dir': os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'data/processed_data/class_data'),
            'n_splits': 5,  # Number of splits for the cross-validation
            }

    n_large = 10000
    util_val_samples = 1000

    train = not True
    show_results = True

    for dataset in datasets:
        # Load synthetic data without methodology
        best_seed = pd.read_csv(os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'results/class_data', 'low_data', dataset, 'vae', 'best_parameters.csv'))['seed'].iloc[0]
        syn_df_low = pd.read_csv(os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'results/class_data', 'low_data', dataset, 'vae', 'seed_' + str(best_seed) + '_gen_data.csv'))

        # Load synthetic data with methodology
        best_seed = pd.read_csv(os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'results/class_data', 'drs', dataset, 'vae', 'best_parameters.csv'))['seed'].iloc[0]
        syn_df_high = pd.read_csv(os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'results/class_data', 'drs', dataset, 'vae', 'seed_' + str(best_seed) + '_gen_data.csv'))

        # Use real data for validation
        real_val_df = pd.read_csv(os.path.join(args['input_dir'], dataset, 'preprocessed_data_m.csv'))  # Validate on actual real data

        cat_cols = []
        for name in real_val_df.columns:
            col = pd.to_numeric(real_val_df[name], errors='coerce')  # Cast to float, to use float.is_integer later
            if len(real_val_df[name].unique()) > 1:
                if col.apply(float.is_integer).all() and len(real_val_df[name].unique()) < 10:
                    cat_cols.append(name)

        real_val_df = sample_cat(real_val_df, cat_cols, util_val_samples)

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
            x_ticks_labels = []
            for idx_to_predict in range(len(cat_cols)):
                target_var = cat_cols[idx_to_predict]
                output_dir = os.path.join(args['output_dir'], dataset, target_var)
                df = pd.read_csv(os.path.join(output_dir, 'classification_metrics.csv'))
                acc = [df.loc[df['case'] == ucase]['acc'].values[0] for ucase in ucases]
                acc_std = [df.loc[df['case'] == ucase]['acc_std'].values[0] for ucase in ucases]

                # Test the acc between the real and synth cases
                test_synth = ttest_ind_from_stats(acc[0], acc_std[0], args['n_splits'], acc[1], acc_std[1], args['n_splits'], equal_var=False, alternative='two-sided')  # Bilateral test: to test if the methodology is better or worse than the low_data case

                if test_synth.pvalue < 0.01:
                    if acc[1] > acc[0]:
                        print(f"Significant ADVANTAGE for methodology for variable {target_var}: low: {acc[0]} ({acc_std[0]}) vs meth:{acc[1]} ({acc_std[1]})")
                    else:
                        print(f"Significant DISADVANTAGE for methodology for variable {target_var}: low: {acc[0]} ({acc_std[0]}) vs meth:{acc[1]} ({acc_std[1]})")
                else:
                    print(f"No significant difference between low_data and methodology cases for variable {target_var}: {acc[0]} ({acc_std[0]}) vs {acc[1]} ({acc_std[1]})")

                diff_synth.append(acc[1] - acc[0])
                x_ticks_labels.append(target_var)
                t = [target_var] + [f"{acc[0]:.3f} ({acc_std[0]:.3f})"] + [f"{acc[1]:.3f} ({acc_std[1]:.3f})"]
                table.append(t)

            plt.hist(diff_synth, bins=10, alpha=0.5, label='diff')
            plt.legend(loc='best')
            plt.savefig(os.path.join(args['output_dir'], dataset, 'diff_synth.png'))

            print(tabulate(table, headers=['Big data', 'Low data'], tablefmt='orgtbl'))


