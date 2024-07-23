import os
import sys
import torch
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from copy import deepcopy
from tabulate import tabulate
from colorama import Fore, Style
from joblib import Parallel, delayed
from scipy.stats import ttest_ind_from_stats

from gen_model.data import sample_cat
from divergence_estimation.utils import store_results
from divergence_estimation.evaluator_analysis import evaluate_set
from utility_validation import evaluate_classification_metrics
from gen_model.data_generation.main_generator import main as main_gen


def train_gen_model(args):
    print(Fore.GREEN + 'Training generative model' + Style.RESET_ALL)
    if args['model'] in ['vae']:
        for seed in range(args['n_seeds']):
            log_name = os.path.join(args['output_dir'], 'seed_' + str(seed))
            args['real_df'].to_csv(log_name + '_real_data.csv', index=False)  # Save real data, for future use
        args['real_df'] = None
        args['n_threads'] = 10  # We do not parallelize the VAE training, although we could (something weird happens with our machine when doing this). Maybe we could move it to GPU...
        main_gen(args)
    else:
        raise RuntimeError('Generative model not recognized')


def validation_method(n, m, l, new_seed, args, cfg):

    # Create folder to store results
    os.makedirs(args['output_dir'] + '/kl', exist_ok=True)
    os.makedirs(args['output_dir'] + '/js', exist_ok=True)
    os.makedirs(args['output_dir'] + '/kl' + f'/{n}_{m}_{l}', exist_ok=True)
    os.makedirs(args['output_dir'] + '/js' + f'/{n}_{m}_{l}', exist_ok=True)

    # Load data
    x_real = args['data_val']
    x_gen = args['syn_df']
    col_names = x_real.columns
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # Dataframe to tensor
    x_real = torch.tensor(x_real.values, dtype=torch.float32, device=device)
    x_gen = torch.tensor(x_gen.values, dtype=torch.float32, device=device)
    # This function calls the divergence estimation methods and saves the results
    evaluate_set(x_real, x_gen, n, m, l, new_seed, args['output_dir'], pre_path=None, case='data', tsne_flag=False,
                 l_gt=None, cfg=cfg, pr=None, ps=None, dataset_name=args['dataset_name'])


def evaluate_gen_model(n, m, l, args):  # Evaluate JS / KL and save results

    seeds = [i for i in range(args['runs'])]

    # Create cfg object with the configuration of the validation method
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10000, type=int, help='Number of epochs to train the discriminator')
    parser.add_argument('--save_model', default=True, type=bool, help='Save the model')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--use_pretrained', default=False, type=bool, help='Whether to use a pretrained model')
    parser.add_argument('--print_feat_js', default=False, type=bool, help='Whether to print the JS per feature')
    parser.add_argument('--name', default=args['dataset_name'], type=str, help='Name of the data')

    cfg = parser.parse_args()

    # Run experiments

    _ = Parallel(n_jobs=args['n_threads'], verbose=10)(delayed(validation_method)(n, m, l, seed, args, cfg)
                                                       for seed in seeds)

    # Create folder to store results
    store_results(args['output_dir'], cfg)


def case_run(n, m, l, separate_training_evaluation, case_name, datasets, gen_methods, args, gen=True, validation=True, methodology=False, methodology_params=None, marginal_plot=True, utility_validation=False, util_val=1000):

    for dataset in datasets:
        # Prepare the data, so that it is the same for all generative models
        train_data = pd.read_csv(os.path.join(args['input_dir'], dataset, 'preprocessed_data_n.csv'))
        metadata_file = open(os.path.join(args['input_dir'], dataset, 'metadata.pkl'), 'rb')
        metadata = pickle.load(metadata_file)
        metadata_file.close()
        cat_cols = [key for key, value in metadata['metadata'].columns.items() if value['sdtype'] == 'categorical']
        train_data = sample_cat(train_data, cat_cols, n)  # Sample data paying attention to having all categories!

        util_val_data = pd.read_csv(os.path.join(args['input_dir'], dataset, 'preprocessed_data_m.csv'))
        util_val_data = sample_cat(util_val_data, cat_cols, util_val)  # Data for utility validation, use a fixed set to have comparable results

        args['real_df'] = train_data
        args['metadata'] = metadata

        data_m = []
        data_l = []

        for j, s in enumerate(separate_training_evaluation):
            if s:  # Separate sets for training and evaluation
                dm = pd.read_csv(os.path.join(args['input_dir'], dataset, 'preprocessed_data_m.csv'))
                dl = pd.read_csv(os.path.join(args['input_dir'], dataset, 'preprocessed_data_l.csv'))
                data_m.append(sample_cat(dm, cat_cols, m[j]))
                data_l.append(sample_cat(dl, cat_cols, 2 * l[j]))
            else:  # The same data is used for training and evaluation
                data_m.append(sample_cat(train_data, cat_cols, m[j]))
                data_l.append(sample_cat(train_data, cat_cols, 2 * l[j]))

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
                train_data = sample_cat(train_data, cat_cols, n)  # Sample data paying attention to having all categories!
                args['real_df'] = train_data
            elif methodology and methodology_params['phase'] == 'pre_train_synth_meta_drs':
                assert gen_method == 'vae', 'Only VAE is supported for meta-learning'
                train_data = []
                for seed in range(args['n_seeds']):
                    train_data.append(pd.read_csv(os.path.join(args['output_dir'], 'low_data', dataset, gen_method, 'seed_' + str(seed) + '_gen_data.csv')))
                print('Replacing real data as generative model input with synthetic data for meta-learning...')
                # Concatenate all the train data
                train_data = pd.concat(train_data, axis=0)  # Build a new dataset using data from all VAE seeds
                train_data = sample_cat(train_data, cat_cols, n)  # Sample data paying attention to having all categories!
                args['real_df'] = train_data
            # First, train the generative model.
            args_gen = deepcopy(args)
            args_gen['dataset_name'] = dataset
            args_gen['model'] = gen_method
            args_gen['case_name'] = case_name
            args_gen['train'] = True
            args_gen['eval'] = True
            args_gen['generated_samples'] = 10000
            args_gen['classifiers_list'] = ['MLP', 'RF']
            args_gen['present_results'] = True
            args_gen['param_comb'] = [{'hidden_size': 256, 'latent_dim': 20}]  # Hyperparameters for the models
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
            elif methodology and methodology_params['phase'] == 'pre_train_synth_meta_drs':
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
            if validation:
                if gen_method == 'vae':  # Evaluate only the best seed
                    best_seed = pd.read_csv(os.path.join(args_gen['output_dir'], 'best_parameters.csv'))['seed'].iloc[0]
                    syn_df = pd.read_csv(os.path.join(args_gen['output_dir'], 'seed_' + str(best_seed) + '_gen_data.csv'))
                else:
                    syn_df = pd.read_csv(os.path.join(args_gen['output_dir'], 'gen_data.csv'))

                for j in range(len(separate_training_evaluation)):
                    args_val['n_threads'] = args['n_threads']
                    args_val['data_val'] = pd.concat([data_m[j], data_l[j]], axis=0)
                    # Now, evaluate the generative model using KL and JS divergences
                    args_val['output_dir'] = os.path.join(args_gen['output_dir'], 'validation')
                    args_val['runs'] = 5  # Number of different KL and JS estimators to use
                    args_val['syn_df'] = sample_cat(syn_df, cat_cols, m[j] + 2 * l[j])  # Sample the synthetic data

                    evaluate_gen_model(n, m[j], l[j], args_val)  # Evaluate and save results

            if utility_validation: # This code is prepared for a classification utility validation (i.e., data are from a classification problem)
                if gen_method == 'vae':  # Evaluate only the best seed
                    best_seed = pd.read_csv(os.path.join(args_gen['output_dir'], 'best_parameters.csv'))['seed'].iloc[0]
                    syn_df = pd.read_csv(os.path.join(args_gen['output_dir'], 'seed_' + str(best_seed) + '_gen_data.csv'))
                else:
                    syn_df = pd.read_csv(os.path.join(args_gen['output_dir'], 'gen_data.csv'))
                args_val['output_dir'] = os.path.join(args_gen['output_dir'], 'utility_validation')
                args_val['syn_df'] = syn_df  # Use the whole generated dataset for utility validation
                args_val['util_val_df'] = util_val_data  # Use the same data for utility validation
                args_val['n_splits'] = args['n_splits']  # Number of splits for the cross-validation (note that we select k-folds for train, but validate using always the same data unseen during training!!)
                evaluate_classification_metrics(args_val)

        if marginal_plot:  # Save the marginal histograms of each feature
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
    datasets = ['3_data', '4_data', '7_data']
    gen_methods = ['vae']
    args = {'n_threads': 10,  # Number of threads for parallelization in the validation phase
            'output_dir': os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'results/class_data'),
            'input_dir': os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'data/processed_data/class_data'),
            'n_seeds': 10,  # Number of seeds for the VAE
            'n_splits': 5,  # Number of splits for the cross-validation
            }
    train_methods = not True  # Flag to train everything
    gen = not True  # To train the generative models or the meta-learner
    validation = not True  # To evaluate the generative models using divergences
    utility_validation = not True  # To evaluate the generative models using a utility metric
    marginal_plot = not True  # To store the marginal plots
    show_results = True  # Flag to show the results

    n_large = 10000
    m_large = 7500
    l_large = 1000
    n_low = 100
    m_low = 75
    l_low = 10

    if train_methods: # Note that, depending on the computational capabilities, as well as the dataset and cases selected, this may take a long time

        # CASE 1: BIG DATA (N=10000, M=7500, L=1000), separate training and evaluation set (ideal conditions)
        # Note that M, L, and separate_training_evaluation are lists, so that we can run several experiments at once (they must correspond to each other)
        case_run(n_large, [m_large], [l_large], [True], 'big_data', datasets, gen_methods, args, methodology=False,
                 gen=gen, validation=validation, marginal_plot=marginal_plot, utility_validation=utility_validation)

        # CASE 2: REALISTIC SITUATION: lower number of samples, and two validations: a realistic one (low number of samples, no separation), and an ideal one (more samples, separation), the latter to evaluate the divergence estimator
        case_run(n_low, [m_large, m_low], [l_large, l_low], [True, False], 'low_data', datasets, gen_methods, args, methodology=False,
                 gen=gen, validation=validation, marginal_plot=marginal_plot, utility_validation=utility_validation)

        # CASE 3: PRETRAINING + TRANSFER LEARNING
        methodology_params = {'phase': 'pre_train_synth'}  # Adjusts everything for pretraining using the case_2 generated data
        case_run(n_large, [m_large, m_low], [l_large, l_low], [True, False], 'pretrain', datasets, gen_methods, args, methodology=True,
                 methodology_params=methodology_params, gen=gen, validation=validation, marginal_plot=marginal_plot, utility_validation=False)

        methodology_params = {'phase': 'fine_tune'}  # Adjusts everything to fine-tune using the pretrained model
        case_run(n_low, [m_large, m_low], [l_large, l_low], [True, False], 'pretrain', datasets, gen_methods, args, methodology=True,
                 methodology_params=methodology_params, gen=gen, validation=validation, marginal_plot=marginal_plot, utility_validation=utility_validation)

        # CASE 4: DRS META LEARNING + TRANSFER LEARNING (only for VAE)
        methodology_params = {'phase': 'pre_train_synth_meta_drs'}  # Adjusts everything for pretraining using the case_2 generated data
        case_run(n_large, [m_low], [l_low], [True], 'drs', datasets, ['vae'], args, methodology=True,
                 methodology_params=methodology_params, gen=gen, validation=False, marginal_plot=False, utility_validation=False)  # Do NOT validate here (non-sense), note that m, l and separate_training_evaluation are not used here

        methodology_params = {'phase': 'fine_tune_meta'}  # Adjusts everything to fine-tune using the pretrained model
        case_run(n_low, [m_large, m_low], [l_large, l_low], [True, False], 'drs', datasets, ['vae'], args, methodology=True,
                 methodology_params=methodology_params, gen=gen, validation=validation, marginal_plot=marginal_plot, utility_validation=utility_validation)

        # CASE 5: MODEL AVERAGING + TRANSFER LEARNING (only for VAE)
        methodology_params = {'phase': 'pre_train_synth_avg'}  # Adjusts everything for pretraining using the case_2 generated data
        case_run(n_large, [m_low], [l_low], [True], 'avg', datasets, ['vae'], args, methodology=True,
                 methodology_params=methodology_params, gen=gen, validation=False, marginal_plot=False, utility_validation=False)  # Do NOT validate here (non-sense), note that m, l and separate_training_evaluation are not used here

        methodology_params = {'phase': 'fine_tune_avg'}  # Adjusts everything to fine-tune using the pretrained model
        case_run(n_low, [m_large, m_low], [l_large, l_low], [True, False], 'avg', datasets, ['vae'], args, methodology=True,
                 methodology_params=methodology_params, gen=gen, validation=validation, marginal_plot=marginal_plot, utility_validation=utility_validation)


    if show_results:  # Show the results obtained
        # Fidelity validation
        print(Fore.GREEN + '\nSimilarity validation' + Style.RESET_ALL)
        per_dataset_metrics = {'pretrain': [], 'drs': [], 'avg': []}
        vae_only_cases = ['drs', 'avg']
        for dataset in datasets:
            print('\n\nResults for dataset ' + Fore.BLUE + dataset + Style.RESET_ALL)
            tab = []
            names = ['Case', 'N', 'M', 'L'] + [f"{gen_method} JS" for gen_method in gen_methods] + [f"{gen_method} KL" for gen_method in gen_methods]
            for case in ['big_data', 'low_data', 'pretrain_s', 'pretrain', 'drs', 'avg']:
                if case != 'pretrain_s':
                    dir = os.path.join(args['output_dir'], case, dataset, gen_methods[0], 'validation')
                else:
                    dir = os.path.join(args['output_dir'], 'pretrain', dataset, gen_methods[0], 'synthetic_pretrain', 'validation')
                d = pd.read_csv(dir + '/js.csv')  # There might be different validation methods
                for i in range(len(d)):
                    # Get N, M and L from the first folder name
                    n, m, l = d['n'].iloc[i], d['m'].iloc[i], d['l'].iloc[i]
                    t = [case, n, m, l]
                    for gen_method in gen_methods:
                        if case in vae_only_cases and gen_method != "vae":
                            t.append("N/A")
                        else:
                            if case != 'pretrain_s':
                                dir = os.path.join(args['output_dir'], case, dataset, gen_method, 'validation')
                            else:
                                dir = os.path.join(args['output_dir'], 'pretrain', dataset, gen_method, 'synthetic_pretrain', 'validation')
                            js = pd.read_csv(os.path.join(dir, 'js.csv'))['JS Discriminator'].iloc[i]
                            js_std = pd.read_csv(os.path.join(dir, 'js_std.csv'))['JS Discriminator'].iloc[i]
                            t.append(f"{js:.3f} ({js_std:.3f})")
                    for gen_method in gen_methods:
                        if case in vae_only_cases and gen_method != "vae":
                            t.append("N/A")
                        else:
                            if case != 'pretrain_s':
                                dir = os.path.join(args['output_dir'], case, dataset, gen_method, 'validation')
                            else:
                                dir = os.path.join(args['output_dir'], 'pretrain', dataset, gen_method, 'synthetic_pretrain', 'validation')
                            kl = pd.read_csv(os.path.join(dir, 'kl.csv'))['KL Discriminator'].iloc[i]
                            kl_std = pd.read_csv(os.path.join(dir, 'kl_std.csv'))['KL Discriminator'].iloc[i]
                            t.append(f"{kl:.3f} ({kl_std:.3f})")
                    tab.append(t)
            # Add the gains observed in case 3
            t = [f"pretrain gain", "-", f"{m_large}", f"{l_large}"]
            js_gain = []
            for gen_method in gen_methods:
                dir = os.path.join(args['output_dir'], 'low_data', dataset, gen_method, 'validation')
                df = pd.read_csv(os.path.join(dir, 'js.csv'))  # Important note: we compute gains over the case with large number of samples
                js_base = df.loc[df['m'] == m_large]['JS Discriminator'].values[0]
                dir = os.path.join(args['output_dir'], 'pretrain', dataset, gen_method, 'validation')
                df = pd.read_csv(os.path.join(dir, 'js.csv'))
                js = df.loc[df['m'] == m_large]['JS Discriminator'].values[0]
                t.append(f"{js_base - js:.3f}")
                js_gain.append(js_base - js)
            per_dataset_metrics['pretrain'].append(js_gain)
            for gen_method in gen_methods:
                dir = os.path.join(args['output_dir'], 'low_data', dataset, gen_method, 'validation')
                df = pd.read_csv(os.path.join(dir, 'kl.csv'))  # Important note: we compute gains over the case with large number of samples
                kl_base = df.loc[df['m'] == m_large]['KL Discriminator'].values[0]
                dir = os.path.join(args['output_dir'], 'pretrain', dataset, gen_method, 'validation')
                df = pd.read_csv(os.path.join(dir, 'kl.csv'))
                kl = df.loc[df['m'] == m_large]['KL Discriminator'].values[0]
                t.append(f"{kl_base - kl:.3f}")
            tab.append(t)

            # Add the gains observed in case 4, 5 and 6: the base metrics are common to the three cases
            dir = os.path.join(args['output_dir'], 'low_data', dataset, 'vae', 'validation')
            df = pd.read_csv(os.path.join(dir, 'js.csv'))
            js_base = df.loc[df['m'] == m_large]['JS Discriminator'].values[0]
            df = pd.read_csv(os.path.join(dir, 'kl.csv'))
            kl_base = df.loc[df['m'] == m_large]['KL Discriminator'].values[0]

            na_string = ["N/A" for gen_method in gen_methods if gen_method != "vae"]

            for case in vae_only_cases:
                dir = os.path.join(args['output_dir'], case, dataset, 'vae', 'validation')
                df = pd.read_csv(os.path.join(dir, 'js.csv'))
                js = df.loc[df['m'] == m_large]['JS Discriminator'].values[0]
                df = pd.read_csv(os.path.join(dir, 'kl.csv'))
                kl = df.loc[df['m'] == m_large]['KL Discriminator'].values[0]
                tab.append(
                    [f"{case} gain", "-", f"{m_large}", f"{l_large}", f"{js_base - js:.3f}"] + na_string + [f"{kl_base - kl:.3f}"]
                    + na_string)
                per_dataset_metrics[case].append(js_base - js)

            print(tabulate(tab, headers=names, tablefmt='orgtbl'))

        print('\n' + Fore.BLUE + 'AVERAGE GAINS' + Style.RESET_ALL)
        print('Average GAIN in pretrain')
        for i, gen_method in enumerate(gen_methods):
            print(f"{gen_method}: {np.mean([d[i] for d in per_dataset_metrics['pretrain']])}")
        for case in vae_only_cases:
            print(f"Average GAIN in case {case} for VAE: {np.mean(per_dataset_metrics[case])}")

        # Utility validation
        print(Fore.GREEN + '\nUtility validation' + Style.RESET_ALL)
        ucases = ['real', 'synth', 'fine_tune']
        per_dataset_metrics = [{'pretrain': [], 'drs': [], 'avg': []} for _ in range(len(ucases))]
        for dataset in datasets:
            print('\n\nResults for dataset ' + Fore.BLUE + dataset + Style.RESET_ALL)
            tab = []
            names = ['Case'] + [f"{ucase} F1" for ucase in ucases] + [f"{ucase} acc" for ucase in ucases]

            acc_base = pd.read_csv(os.path.join(args['output_dir'], 'low_data', dataset, 'vae', 'utility_validation',
                                                'classification_metrics.csv'))['acc'][0]
            acc_std_base = pd.read_csv(
                os.path.join(args['output_dir'], 'low_data', dataset, 'vae', 'utility_validation',
                             'classification_metrics.csv'))['acc_std'][0]

            for case in ['big_data', 'low_data', 'pretrain', 'drs', 'avg']:
                df = pd.read_csv(os.path.join(args['output_dir'], case, dataset, 'vae', 'utility_validation', 'classification_metrics.csv'))
                f1 = [df.loc[df['case'] == ucase]['f1'].values[0] for ucase in ucases]
                f1_std = [df.loc[df['case'] == ucase]['f1_std'].values[0] for ucase in ucases]
                acc = [df.loc[df['case'] == ucase]['acc'].values[0] for ucase in ucases]
                acc_std = [df.loc[df['case'] == ucase]['acc_std'].values[0] for ucase in ucases]
                t = [case] + [f"{f1[i]:.3f} ({f1_std[i]:.3f})" for i in range(len(ucases))] + [f"{acc[i]:.3f} ({acc_std[i]:.3f})" for i in range(len(ucases))]
                tab.append(t)
                for i in range(len(ucases)):
                    if case != 'big_data':
                        acc_, acc_std_ = acc[i], acc_std[i]
                        test = ttest_ind_from_stats(acc_base, acc_std_base, args['n_splits'], acc_, acc_std_, args['n_splits'], equal_var=False, alternative='two-sided')
                        if test.pvalue < 0.01:
                            if acc_base < acc_:
                                print(
                                    f"Significant ADVANTAGE for methodology {case}: low_data: {acc_base} ({acc_std_base}) vs meth:{acc_} ({acc_std_})")
                            else:
                                print(f"Significant DISADVANTAGE for methodology {case}: low_data: {acc_base} ({acc_std_base}) vs meth:{acc_} ({acc_std_})")
                        else:
                            print(f"No significant difference for methodology {case}: low_data: {acc_base} ({acc_std_base}) vs meth:{acc_} ({acc_std_})")

            # Add the gains observed in case 3
            acc_gain = []
            f1_gain = []
            df_base = pd.read_csv(os.path.join(args['output_dir'], 'low_data', dataset, 'vae', 'utility_validation', 'classification_metrics.csv'))
            f1_base = [df_base.loc[df_base['case'] == ucase]['f1'].values[0] for ucase in ucases]
            acc_base = [df_base.loc[df_base['case'] == ucase]['acc'].values[0] for ucase in ucases]

            for case in ['pretrain', 'avg', 'drs']:
                df = pd.read_csv(os.path.join(args['output_dir'], case, dataset, 'vae', 'utility_validation', 'classification_metrics.csv'))
                f1 = [df.loc[df['case'] == ucase]['f1'].values[0] for ucase in ucases]
                acc = [df.loc[df['case'] == ucase]['acc'].values[0] for ucase in ucases]
                t = [f"{case} gain"] + [f"{f1[i] - f1_base[i]:.3f}" for i in range(len(ucases))] + [f"{acc[i] - acc_base[i]:.3f}" for i in range(len(ucases))]
                tab.append(t)
                for i in range(len(ucases)):
                    per_dataset_metrics[i][case].append(acc[i] - acc_base[i])

            print('\n')
            print(tabulate(tab, headers=names, tablefmt='orgtbl'))

        print('\n' + Fore.BLUE + 'AVERAGE GAINS in pretrain' + Style.RESET_ALL)
        for i in range(len(ucases)):
            for case in ['pretrain', 'drs', 'avg']:
                print(f"{ucases[i]}/{case}: {np.mean(per_dataset_metrics[i][case])}")