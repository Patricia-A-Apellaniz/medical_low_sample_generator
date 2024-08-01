import os
import sys
import torch
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from copy import deepcopy
from tabulate import tabulate
from colorama import Fore, Style
from scipy.stats import ttest_ind_from_stats
from utility_validation import evaluate_survival_metrics
from gen_model.data_generation.main_generator import main as main_gen


def train_gen_model(args):
    print(Fore.GREEN + 'Training generative model' + Style.RESET_ALL)
    if args['model'] in ['vae']:
        for seed in range(args['n_seeds']):
            log_name = os.path.join(args['output_dir'], 'seed_' + str(seed))
            args['real_df'].to_csv(log_name + '_real_data.csv', index=False)  # Save real data, for future use
        args['real_df'] = None
        args['n_threads'] = 1  # We do not parallelize the VAE training, although we could (something weird happens with our machine when doing this). Maybe we could move it to GPU...
        main_gen(args)
    else:
        raise RuntimeError('Generative model not recognized')


def case_run(case_name, datasets, gen_methods, args, gen=True, utility_validation=True, methodology=False, methodology_params=None, marginal_plot=True):

    for dataset in datasets:
        # Prepare the data, so that it is the same for all generative models
        train_data = pd.read_csv(os.path.join(args['input_dir'], dataset, 'preprocessed_data_all.csv'))
        # Use always a 20% of the data for validation
        util_val_data = train_data.sample(frac=0.2, random_state=0)  # Data used for validation, always the same size, for comparison among all methods
        train_data = train_data.drop(util_val_data.index)
        if case_name != 'big_data':  # If we are not in the big data case, then use only the first 100 samples
            train_data = train_data.iloc[:100]

        metadata_file = open(os.path.join(args['input_dir'], 'processed_data', dataset, 'metadata.pkl'), 'rb')
        metadata = pickle.load(metadata_file)
        metadata_file.close()

        args['real_df'] = train_data
        args['metadata'] = metadata

        for gen_method in gen_methods:
            # If we apply the methodology proposed, and we want to pretrain, reload the synthetic data here
            if methodology and methodology_params['phase'] == 'pre_train_synth':
                gen_data_dir = os.path.join(args['output_dir'], 'low_data', dataset, gen_method)
                if gen_method == 'vae':  # Load the data from the best seed only
                    best_seed = pd.read_csv(os.path.join(gen_data_dir, 'best_parameters.csv'))['seed'].iloc[0]
                    train_data = pd.read_csv(os.path.join(gen_data_dir, 'seed_' + str(best_seed) + '_gen_data.csv'))
                else:
                    train_data = pd.read_csv(os.path.join(gen_data_dir, 'gen_data.csv'))
                print('Replacing real data as generative model input with synthetic data...')
                args['real_df'] = train_data
            elif methodology and methodology_params['phase'] == 'pre_train_synth_meta':
                assert gen_method == 'vae', 'Only VAE is supported for meta-learning'
                train_data = []
                for seed in range(args['n_seeds']):
                    train_data.append(pd.read_csv(os.path.join(args['output_dir'], 'low_data', dataset, gen_method, 'seed_' + str(seed) + '_gen_data.csv')))
                print('Replacing real data as generative model input with synthetic data for meta-learning...')
                args['real_df'] = train_data
            elif methodology and methodology_params['phase'] == 'pre_train_synth_meta_drs':
                assert gen_method == 'vae', 'Only VAE is supported for meta-learning'
                train_data = []
                for seed in range(args['n_seeds']):
                    train_data.append(pd.read_csv(os.path.join(args['output_dir'], 'low_data', dataset, gen_method, 'seed_' + str(seed) + '_gen_data.csv')))
                print('Replacing real data as generative model input with synthetic data for meta-learning...')
                # Concatenate all the train data
                train_data = pd.concat(train_data, axis=0)  # Build a new dataset using data from all VAE seeds
                args['real_df'] = train_data
            # First, train the generative model.
            args_gen = deepcopy(args)
            args_gen['dataset_name'] = dataset
            args_gen['model'] = gen_method
            args_gen['case_name'] = case_name
            args_gen['train'] = True
            args_gen['generated_samples'] = 10000
            args_gen['param_comb'] = [{'hidden_size': 256, 'latent_dim': 10}]  # Hyperparameters for the models
            if gen_method in ['vae']:  # VAE parameters
                args_gen['imp_mask'] = False  # We assume all data has been imputed during preprocessing
                args_gen['mask_gen'] = False
                args_gen['train_vae'] = True
                args_gen['early_stop'] = True
                args_gen['batch_size'] = 256

            if methodology and methodology_params['phase'] == 'fine_tune':  # If we are to fine-tune, then load the model
                args_gen['use_pretrained'] = True
                args_gen['pretrained_dir'] = os.path.join(args['output_dir'], case_name, dataset, gen_method, 'synthetic_pretrain')
            elif methodology and methodology_params['phase'] == 'fine_tune_meta':  # If we are to fine-tune, then load the model
                args_gen['use_pretrained'] = True
                args_gen['pretrained_dir'] = os.path.join(args['output_dir'], case_name, dataset, gen_method, 'meta_pretrain')
            elif methodology and methodology_params['phase'] == 'fine_tune_avg':  # If we are to fine-tune, then load the model
                args_gen['use_pretrained'] = True
                args_gen['pretrained_dir'] = os.path.join(args['output_dir'], case_name, dataset, gen_method, 'avg_pretrain')
            else:
                args_gen['use_pretrained'] = False
                args_gen['pretrained_dir'] = None

            if methodology and methodology_params['phase'] == 'pre_train_synth':  # If we are to pretrain, then save the models in a different folder
                args_gen['output_dir'] = os.path.join(args['output_dir'], case_name, dataset, gen_method, 'synthetic_pretrain')
            elif methodology and (methodology_params['phase'] == 'pre_train_synth_meta' or methodology_params['phase'] == 'pre_train_synth_meta_drs'):
                args_gen['output_dir'] = os.path.join(args['output_dir'], case_name, dataset, gen_method, 'meta_pretrain')
            elif methodology and methodology_params['phase'] == 'pre_train_synth_avg':
                args_gen['output_dir'] = os.path.join(args['output_dir'], case_name, dataset, gen_method, 'avg_pretrain')
            else:
                args_gen['output_dir'] = os.path.join(args['output_dir'], case_name, dataset, gen_method)
            os.makedirs(args_gen['output_dir'], exist_ok=True)  # Ensure that the path exists

            args_val = deepcopy(args_gen)  # Copy the arguments for the validation phase

            if gen and methodology and methodology_params['phase'] == 'pre_train_synth_meta_drs':  # In this case, train the DRS meta-learner
                args_gen['n_seeds'] = 1  # Generate a single seed for the DRS pretraining
                train_gen_model(args_gen)  # DRS is a "standard" pretrain
            elif gen and methodology and methodology_params['phase'] == 'pre_train_synth_avg':  # In this case, we just compute an average model
                models = [torch.load(os.path.join(args['output_dir'], 'low_data', dataset, gen_method, 'seed_' + str(seed))) for seed in range(args_gen['n_seeds'])]
                new_model = deepcopy(models[0])
                for key in new_model.keys():
                    avg_val = sum([m[key] for m in models]) / args_gen['n_seeds']
                    new_model[key] = avg_val
                torch.save(new_model, os.path.join(args_gen['output_dir'], 'avg_pretrain'))
            elif gen: # Note that the parameters may be used by the evaluation method
                train_gen_model(args_gen)  # Train and save results

            # Prepare for evaluation (there might be several situations to validate!)
            if utility_validation:  # This code is prepared for a classification utility validation (i.e., data are from a classification problem)
                if gen_method == 'vae':  # Evaluate only the best seed
                    best_seed = pd.read_csv(os.path.join(args_gen['output_dir'], 'best_parameters.csv'))['seed'].iloc[0]
                    syn_df = pd.read_csv(os.path.join(args_gen['output_dir'], 'seed_' + str(best_seed) + '_gen_data.csv'))
                else:
                    syn_df = pd.read_csv(os.path.join(args_gen['output_dir'], 'gen_data.csv'))
                args_val['output_dir'] = os.path.join(args_gen['output_dir'], 'utility_validation')
                args_val['syn_df'] = syn_df  # Use the whole generated dataset for utility validation
                args_val['n_splits'] = args['n_splits']  # Number of splits for the cross-validation
                args_val['util_val_df'] = util_val_data  # Use the same data for utility validation
                evaluate_survival_metrics(args_val)

        if marginal_plot:  # Save the marginal histograms of each feature
            if methodology and methodology_params['phase'] == 'pre_train_synth_meta':
                pass  # Nothing to plot in this case
            else:
                dfs = [args['real_df']]
                names = ['real']
                for gen_method in gen_methods:
                    if methodology and methodology_params['phase'] == 'pre_train_synth':  # If we are to pretrain, then save the models in a different folder
                        output_dir = os.path.join(args['output_dir'], case_name, dataset, gen_method, 'synthetic_pretrain')
                    else:
                        output_dir = os.path.join(args['output_dir'], case_name, dataset, gen_method)
                    if gen_method == 'vae':  # Evaluate only the best seed
                        best_seed = pd.read_csv(os.path.join(output_dir, 'best_parameters.csv'))['seed'].iloc[0]
                        syn_df = pd.read_csv(os.path.join(output_dir, 'seed_' + str(best_seed) + '_gen_data.csv'))
                    else:
                        syn_df = pd.read_csv(os.path.join(output_dir, 'gen_data.csv'))
                    dfs.append(syn_df)
                    names.append(gen_method)
                for lab in args['real_df'].columns.tolist():
                    plt.hist([df[lab] for df in dfs], bins=10, label=names, density=True)
                    plt.title(f'Marginal distribution of {lab} in {dataset}')
                    plt.legend(loc='best')
                    plt.savefig(os.path.join(args['output_dir'], case_name, dataset, 'marginal_' + lab + '.png'))
                    plt.close()


if __name__ == '__main__':
    ## MAIN PARAMETERS OF THE CODE: CHANGE THINGS HERE
    datasets = ['metabric', 'gbsg', 'whas', 'nwtco', 'pbc', 'std']
    gen_methods = ['vae']
    args = {'n_threads': 10,  # Number of threads for parallelization in the validation phase
            'output_dir': os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'results/sa_data'),
            'input_dir': os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'data/processed_data/sa_data'),
            'n_seeds': 10,  # Number of seeds for the VAE
            'n_splits': 5,  # Number of splits for the cross-validation
            }
    train_methods = not True  # Flag to train everything
    gen = not True  # To train the generative models or the meta-learner
    utility_validation = not True  # To evaluate the generative models
    marginal_plot = not True  # To store the marginal plots
    show_results = True  # Flag to show the results

    if train_methods: # Note that, depending on the computational capabilities, as well as the dataset and cases selected, this may take a long time

        # REAL SITUATION: actual number of samples
        case_run('big_data', datasets, gen_methods, args, methodology=False, gen=gen, utility_validation=utility_validation, marginal_plot=marginal_plot)

        # LOW DATA SITUATION: 100 samples
        case_run('low_data', datasets, gen_methods, args, methodology=False, gen=gen, utility_validation=utility_validation, marginal_plot=marginal_plot)

        # CASE 3: PRETRAINING + TRANSFER LEARNING
        methodology_params = {'phase': 'pre_train_synth'}  # Adjusts everything for pretraining using the case_2 generated data
        case_run('pretrain', datasets, gen_methods, args, methodology=True,
                 methodology_params=methodology_params, gen=gen, utility_validation=False, marginal_plot=marginal_plot)

        methodology_params = {'phase': 'fine_tune'}  # Adjusts everything to fine-tune using the pretrained model
        case_run('pretrain', datasets, gen_methods, args, methodology=True,
                 methodology_params=methodology_params, gen=gen, utility_validation=utility_validation, marginal_plot=marginal_plot)

        # CASE 4: DRS META LEARNING + TRANSFER LEARNING (only for VAE)
        methodology_params = {'phase': 'pre_train_synth_meta_drs'}  # Adjusts everything for pretraining using the case_2 generated data
        case_run('drs', datasets, ['vae'], args, methodology=True,
                 methodology_params=methodology_params, gen=gen, utility_validation=False, marginal_plot=False)  # Do NOT validate here (non-sense), note that m, l and separate_training_evaluation are not used here

        methodology_params = {'phase': 'fine_tune_meta'}  # Adjusts everything to fine-tune using the pretrained model
        case_run('drs', datasets, ['vae'], args, methodology=True,
                 methodology_params=methodology_params, gen=gen, utility_validation=utility_validation, marginal_plot=marginal_plot)

        # CASE 5: MODEL AVERAGING + TRANSFER LEARNING (only for VAE)
        methodology_params = {'phase': 'pre_train_synth_avg'}  # Adjusts everything for pretraining using the case_2 generated data
        case_run('avg', datasets, ['vae'], args, methodology=True,
                 methodology_params=methodology_params, gen=gen, utility_validation=False, marginal_plot=False)  # Do NOT validate here (non-sense), note that m, l and separate_training_evaluation are not used here

        methodology_params = {'phase': 'fine_tune_avg'}  # Adjusts everything to fine-tune using the pretrained model
        case_run('avg', datasets, ['vae'], args, methodology=True,
                 methodology_params=methodology_params, gen=gen, utility_validation=utility_validation, marginal_plot=marginal_plot)


    if show_results:  # Show the results obtained
        # Fidelity validation: not done in this case, as the low number of samples prevents having reliable numbers!
        # Utility validation
        print(Fore.GREEN + '\nUtility validation' + Style.RESET_ALL)
        ucases = ['real', 'synth', 'fine_tune']
        per_dataset_metrics = [{'pretrain': [], 'drs': [], 'avg': []} for _ in range(len(ucases))]
        for dataset in datasets:
            print('\n\nResults for dataset ' + Fore.BLUE + dataset + Style.RESET_ALL)
            tab = []
            names = ['Case'] + [f"{ucase} CI" for ucase in ucases] + [f"{ucase} IBS" for ucase in ucases]

            ci_base = pd.read_csv(os.path.join(args['output_dir'], 'low_data', dataset, 'vae', 'utility_validation',
                                                'survival_metrics.csv'))['ci'][0]
            ci_std_base = pd.read_csv(
                os.path.join(args['output_dir'], 'low_data', dataset, 'vae', 'utility_validation',
                             'survival_metrics.csv'))['ci_std'][0]
            ibs_base = pd.read_csv(os.path.join(args['output_dir'], 'low_data', dataset, 'vae', 'utility_validation',
                                               'survival_metrics.csv'))['ibs'][0]
            ibs_std_base = pd.read_csv(
                os.path.join(args['output_dir'], 'low_data', dataset, 'vae', 'utility_validation',
                             'survival_metrics.csv'))['ibs_std'][0]

            for case in ['big_data', 'low_data', 'pretrain', 'drs', 'avg']:
                df = pd.read_csv(os.path.join(args['output_dir'], case, dataset, 'vae', 'utility_validation', 'survival_metrics.csv'))
                ci = [df.loc[df['case'] == ucase]['ci'].values[0] for ucase in ucases]
                ci_std = [df.loc[df['case'] == ucase]['ci_std'].values[0] for ucase in ucases]
                ibs = [df.loc[df['case'] == ucase]['ibs'].values[0] for ucase in ucases]
                ibs_std = [df.loc[df['case'] == ucase]['ibs_std'].values[0] for ucase in ucases]
                t = [case] + [f"{ci[i]:.3f} ({ci_std[i]:.3f})" for i in range(len(ucases))] + [f"{ibs[i]:.3f} ({ibs_std[i]:.3f})" for i in range(len(ucases))]
                tab.append(t)
                for i in range(len(ucases)):
                    if case != 'big_data':
                        ci_, ci_std_, ibs_, ibs_std_ = ci[i], ci_std[i], ibs[i], ibs_std[i]
                        test_ci = ttest_ind_from_stats(ci_base, ci_std_base, args['n_splits'], ci_, ci_std_, args['n_splits'], equal_var=False, alternative='two-sided')
                        test_ibs = ttest_ind_from_stats(ibs_base, ibs_std_base, args['n_splits'], ibs_, ibs_std_, args['n_splits'], equal_var=False, alternative='two-sided')
                        if test_ci.pvalue < 0.01:
                            if ci_base < ci_:
                                print(f"Significant CI ADVANTAGE for methodology {case}/{ucases[i]}: low_data: {ci_base} ({ci_std_base}) vs meth:{ci_} ({ci_std_})")
                            else:
                                print(f"Significant CI DISADVANTAGE for methodology {case}/{ucases[i]}: low_data: {ci_base} ({ci_std_base}) vs meth:{ci_} ({ci_std_})")
                        else:
                            print(f"No significant CI difference for methodology {case}/{ucases[i]}: low_data: {ci_base} ({ci_std_base}) vs meth:{ci_} ({ci_std_})")

                        if test_ibs.pvalue < 0.01:
                            if ibs_base > ibs_:
                                print(f"Significant IBS ADVANTAGE for methodology {case}/{ucases[i]}: low_data: {ibs_base} ({ibs_std_base}) vs meth:{ibs_} ({ibs_std_})")
                            else:
                                print(
                                    f"Significant IBS DISADVANTAGE for methodology {case}/{ucases[i]}: low_data: {ibs_base} ({ibs_std_base}) vs meth:{ibs_} ({ibs_std_})")
                        else:
                            print(f"No significant IBS difference for methodology {case}/{ucases[i]}: low_data: {ibs_base} ({ibs_std_base}) vs meth:{ibs_} ({ibs_std_})")

            # Add the gains observed in case 3
            acc_gain = []
            f1_gain = []
            df_base = pd.read_csv(os.path.join(args['output_dir'], 'low_data', dataset, 'vae', 'utility_validation', 'survival_metrics.csv'))
            ci_base = [df_base.loc[df_base['case'] == ucase]['ci'].values[0] for ucase in ucases]
            ibs_base = [df_base.loc[df_base['case'] == ucase]['ibs'].values[0] for ucase in ucases]

            for case in ['pretrain', 'avg', 'drs']:
                df = pd.read_csv(os.path.join(args['output_dir'], case, dataset, 'vae', 'utility_validation', 'survival_metrics.csv'))
                ci = [df.loc[df['case'] == ucase]['ci'].values[0] for ucase in ucases]
                ibs = [df.loc[df['case'] == ucase]['ibs'].values[0] for ucase in ucases]
                t = [f"{case} gain"] + [f"{ci[i] - ci_base[i]:.3f}" for i in range(len(ucases))] + [f"{ibs_base[i] - ibs[i]:.3f}" for i in range(len(ucases))]
                tab.append(t)
                for i in range(len(ucases)):
                    per_dataset_metrics[i][case].append(ci[i] - ci_base[i])

            print('\n')
            print(tabulate(tab, headers=names, tablefmt='orgtbl'))
            # print(tabulate(tab, headers=names, tablefmt='latex'))

        print('\n' + Fore.BLUE + 'AVERAGE GAINS in pretrain' + Style.RESET_ALL)
        for i in range(len(ucases)):
            for case in ['pretrain', 'drs', 'avg']:
                print(f"{ucases[i]}/{case}: {np.mean(per_dataset_metrics[i][case])}")
