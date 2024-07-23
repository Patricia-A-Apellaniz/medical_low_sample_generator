import os
import torch

import numpy as np
import pandas as pd

from .divergence import KL, JS


class DivergenceEvaluator:
    def __init__(self, x_p, x_q, id, result_path, pre_path, n, m, l, verbose=True, pr=None, ps=None, dataset_name=None):
        self.x_p = x_p
        self.x_q = x_q
        self.id = id
        self.result_path = result_path
        self.pre_path = pre_path
        self.n = n
        self.m = m
        self.l = l
        self.verbose = verbose
        self.pr = pr
        self.ps = ps
        self.dataset_name = dataset_name

    def compute_mc_gt(self, gt_runs=100, l_gt=50000, case='mvn'):
        # Compute ground truth if distribution is known. Only for synthetic data.
        # Monte Carlo estimation is used with a large number of samples and several realisations.
        if self.pr is None or self.ps is None:
            print('Error: pr or ps is None.')
            return None, None

        # Check if ground truth already exists
        gt_folder = self.result_path + '/kl' + f'/{self.n}_{self.m}_{l_gt}'
        # check if the results path exists and is not empty
        if os.path.exists(gt_folder) and os.listdir(gt_folder):
            print('Ground truth already exists.')
            return None, None
        # Define KL, JS object
        kl = KL(self.x_p, self.x_q, fraction_train=1)
        js = JS(self.x_p, self.x_q, fraction_train=1)

        kl_gt = torch.tensor(0)
        js_gt = torch.tensor(0)
        for i in range(1, gt_runs+1):
            # Sample from p and q
            p_test = self.pr.sample((l_gt,))
            if case == 'mvn':
                q_test = self.ps.sample((l_gt,))
            elif 'gm_dif' in case:
                q_test = self.ps.sample((l_gt,))
                q_test = torch.tensor(q_test, dtype=torch.float32)
            elif 'gmm' in case:
                q_test, _ = self.ps.sample(l_gt)
                q_test = torch.tensor(q_test, dtype=torch.float32)
            # Compute KL and JS divergence using Monte Carlo
            kl1 = kl.mc(self.pr, self.ps, p_test=p_test)
            js1 = js.mc(self.pr, self.ps, x_real=p_test, x_gen=q_test)
            # Update mean
            kl_gt =(kl_gt * (i - 1) + kl1) / (i)
            js_gt = (js_gt * (i - 1) + js1) / (i)

        return kl_gt, js_gt

    def evaluate(self, mc, ratio, disc, cfg=None):
        # Compute divergence using different methods.
        M = self.m / (self.m + (2 * self.l)) # M is the fraction of the training set used to train the discriminator.
        n_fit_epochs = cfg.epochs # Number of epochs used to train the discriminator.

        # Define KL, JS object
        kl = KL(self.x_p, self.x_q, M, seed=self.id, pre_path=self.pre_path, results_path=self.result_path, n=self.n,
                device=self.x_q.device, m=self.m, l=self.l, dataset_name=self.dataset_name, cfg=cfg)
        js = JS(self.x_p, self.x_q, M, seed=self.id, pre_path=self.pre_path, results_path=self.result_path, n=self.n,
                device=self.x_q.device, m=self.m, l=self.l, dataset_name=self.dataset_name, cfg=cfg)
        kl1, kl2, kl4, js1, js2, js4 = None, None, None, None, None, None
        # Monte Carlo
        if mc:
            # check if pr is not none, else error or warning
            if self.pr is not None and self.ps is not None:
                kl1 = kl.mc(self.pr, self.ps)
                js1 = js.mc(self.pr, self.ps)
                if self.verbose:
                    print(f'KL divergence via Monte Carlo {np.round(kl1, 4)}')
                    print(f'JS divergence via Monte Carlo {np.round(js1, 4)}')
            else:
                print('ERROR: pr and ps are None, cannot compute MC')

        else:
            kl1 = torch.tensor(0)
            js1 = torch.tensor(0)

        # Perfect ratio
        if ratio:
            if self.pr is not None and self.ps is not None:
                kl2 = kl.perfect_ratio(self.pr, self.ps)
                js2 = js.perfect_ratio(self.pr, self.ps)
                if self.verbose:
                    print(f'KL divergence via r = p/q {np.round(kl2, 4)}')
                    print(f'JS divergence via r = p/q {np.round(js2, 4)}')
                else:
                    print('ERROR: pr and ps are None, cannot compute perfect ratio')
        else:
            kl2 = torch.tensor(0)
            js2 = torch.tensor(0)
        # Discriminator
        if disc:
            path = self.result_path + '/kl' + f'/{self.n}_{self.m}_{self.l}/seed_{self.id}_{self.dataset_name}_model.pt'
            kl4 = kl.forward(n_fit_epochs=n_fit_epochs, n=self.n, save_model=cfg.save_model, path=path, cfg=cfg)
            js4 = js.forward(n_fit_epochs=n_fit_epochs, n=self.n, cfg=cfg)
            kl4 = kl4.cpu().detach()
            js4 = js4.cpu().detach()
            if self.verbose:
                print(f'KL divergence via Discriminator {np.round(kl4.numpy(), 4)}')
                print(f'JS divergence via Discriminator {np.round(js4.numpy(), 4)}')
        else:
            kl4 = torch.tensor(0)
            js4 = torch.tensor(0)

        return kl1, kl2, kl4, js1, js2, js4

    def save_results(self, kl1, kl2, kl3, js1, js2, js3, results_path, case, m, l, n, new_seed, kl_real=torch.tensor(0)):
        # Save results to csv file.
        if case == 'mvn':
            # Create dataframe
            df_kl = pd.DataFrame({'n': n, 'm': m, 'l': l, 'KL MC': kl1.numpy(), 'KL real': kl_real.numpy(),
                                  'KL Discriminator': kl3.numpy()}, index=[0])
            # save to csv
            df_kl.to_csv(results_path + '/kl' + f'/{n}_{m}_{l}/seed_{new_seed}_{self.dataset_name}.csv', index=False)

            # Create dataframe
            df_js = pd.DataFrame({'n': n, 'm': m, 'l': l, 'JS MC': js1.numpy(), 'JS p/q': js2.numpy(),
                                  'JS Discriminator': js3.numpy()}, index=[0])
            # save to csv
            df_js.to_csv(results_path + '/js' + f'/{n}_{m}_{l}/seed_{new_seed}_{self.dataset_name}.csv', index=False)
        elif case == 'gm_dif' or case == 'gmm' or 'data' in case:
            # Create dataframe
            df_kl = {'n': n, 'm': m, 'l': l, 'KL MC': kl1.numpy(), 'KL p/q': kl2.numpy(), 'KL Discriminator': kl3.numpy()}
            df_kl = pd.DataFrame(df_kl, index=[0])
            # save to csv
            df_kl.to_csv(results_path + '/kl' + f'/{n}_{m}_{l}/seed_{new_seed}_{self.dataset_name}.csv', index=False)

            # Create dataframe
            df_js = {'n': n, 'm': m, 'l': l, 'JS MC': js1.numpy(), 'JS p/q': js2.numpy(), 'JS Discriminator': js3.numpy()}
            df_js = pd.DataFrame(df_js, index=[0])
            # save to csv
            df_js.to_csv(results_path + '/js' + f'/{n}_{m}_{l}/seed_{new_seed}_{self.dataset_name}.csv', index=False)

