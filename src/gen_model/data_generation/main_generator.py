# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 12/09/2023

# Packages to import
import os
import pickle
import time

import pandas as pd
import torch
import numpy as np
from colorama import Fore, Style
from ..data import split_data
from ..data_generation.generator import Generator

from joblib import Parallel, delayed


def get_pretrained_vae(args_gen):
    if os.path.exists(os.path.join(args_gen['pretrained_dir'], 'best_parameters.csv')):
        best_seed = pd.read_csv(os.path.join(args_gen['pretrained_dir'], 'best_parameters.csv'))['seed'].iloc[0]
        gen_data_file = args_gen['pretrained_dir'] + '/seed_' + str(best_seed)
    elif 'avg_pretrain' in args_gen['pretrained_dir']:
        gen_data_file = args_gen['pretrained_dir'] + '/avg_pretrain'
    else:
        gen_data_file = args_gen['pretrained_dir'] + '/maml_model'
    return gen_data_file


def train(params, seed, args, dataset_name=None):
    # Load data
    feat_distributions = args['metadata']['feat_distributions']
    # The mask is a pandas dataframe with the same shape as imp_norm_df made of ones, as the data is already imputed in preprocessing.
    log_name = os.path.join(args['output_dir'], 'seed_' + str(seed))
    real_df = pd.read_csv(log_name + '_real_data.csv')
    mask = pd.DataFrame(np.ones(real_df.shape), columns=real_df.columns)
    data = split_data(real_df, mask)

    # Model parameters
    latent_dim = params['latent_dim']
    hidden_size = params['hidden_size']
    input_dim = data[0].shape[1]
    model_params = {'feat_distributions': feat_distributions, 'latent_dim': latent_dim, 'hidden_size': hidden_size,
                    'input_dim': input_dim}
    model = Generator(model_params)

    # Train the base_model
    vae_tr_time = 0
    if args['train_vae']:
        # Note: The number of epochs is large on purpose: there is an early stopper on the model
        device = torch.device('cpu')  # Note: we use CPU for multi-threading, it is usually enough
        print(Fore.GREEN + 'Device: ' + str(device) + Style.RESET_ALL)
        train_params = {'n_epochs': 10000, 'batch_size': args['batch_size'], 'device': device, 'lr': 1e-3,
                        'path_name': log_name}
        if args['use_pretrained']:
            pretrained_vae = get_pretrained_vae(args)
            print(Fore.GREEN + 'Loading pretrained VAE: ' + pretrained_vae + Style.RESET_ALL)

            model.load_state_dict(torch.load(pretrained_vae))
        print(Fore.GREEN + 'Training VAE with parameters: ' + str(train_params) + Style.RESET_ALL)
        start_time = time.time()
        training_results = model.fit(data, train_params)
        vae_tr_time = time.time() - start_time

        # Save base_model information
        model.save(log_name)
        model_params.update(train_params)
        model_params.update(training_results)
        with open(log_name + '.pickle', 'wb') as handle:
            pickle.dump(model_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        # Load already trained model
        model.load_state_dict(torch.load(log_name))
    # Obtain and save synthetic samples using training data
    n_gen = args['generated_samples']
    # Print number of samples to generate
    print(Fore.GREEN + 'Number of samples to generate: ' + str(n_gen) + Style.RESET_ALL)
    # Print seed
    print(Fore.GREEN + 'Seed: ' + str(seed) + Style.RESET_ALL)
    # Print hidden dim and latent size
    print(Fore.GREEN + 'Hidden dim: ' + str(hidden_size) + Style.RESET_ALL)
    print(Fore.GREEN + 'Latent size: ' + str(latent_dim) + Style.RESET_ALL)

    start_time = time.time()
    model.train_latent_generator(data[0])
    gen_data = model.generate(n_gen=n_gen)

    gen_df = pd.DataFrame(gen_data['cov_samples'], columns=real_df.columns.tolist())

    gen_time = time.time() - start_time
    # Print length of generated data
    print(Fore.GREEN + 'Length of generated data: ' + str(len(gen_df)) + 'Seed' + str(seed) + Style.RESET_ALL)
    gen_df.to_csv(log_name + '_gen_data.csv', index=False)

    return vae_tr_time, gen_time, False


def main(args=None):
    print('\n\n-------- SYNTHETIC DATA GENERATION - MODEL --------')

    dataset_name = args['dataset_name']
    print('\n\nDataset: ' + Fore.CYAN + dataset_name + Style.RESET_ALL)
    # Train model
    if args['train']:  # Note: the backend loky seems to be slower than threading
        timing = Parallel(n_jobs=args['n_threads'], verbose=10, backend='threading')(
            delayed(train)(params, seed, args, dataset_name=dataset_name)
            for params in args['param_comb'] for seed in range(args['n_seeds']))
        training_times = list(zip(*timing))[0]
        generating_times = list(zip(*timing))[1]

        print('Average training time: ' + str(format(sum(training_times) / len(training_times), '.2f')) + ' seconds')
        print('Average generating time: ' + str(
            format(sum(generating_times) / len(generating_times), '.2f')) + ' seconds')
    # Evaluation: select as best seed the one with lowest validation loss
    results = []
    for seed in range(args['n_seeds']):
        with open(os.path.join(args['output_dir'], 'seed_' + str(seed) + '.pickle'), 'rb') as handle:
            results.append(pickle.load(handle))
    val_loss = [r['loss_va'][-1] for r in results]
    best_seeds = np.argsort(val_loss)
    parameters2save = {'seed': [best_seeds[0]], 'params': [
        str(args['param_comb'][0]['latent_dim']) + '_' + str(args['param_comb'][0]['hidden_size'])]}
    # dictionary to dataframe
    param_df = pd.DataFrame(parameters2save)
    # save dataframe
    param_df.to_csv(args['output_dir'] + '/best_parameters.csv', index=False)
