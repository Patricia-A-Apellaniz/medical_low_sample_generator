import os
import sys
import torch

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.utils import shuffle
from abc import abstractmethod, ABC
from torch import distributions as D
from .discriminator import Discriminator
from .tabular import tensor_to_dataloader
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def ml_comparison(p_samples, q_samples, p_eval, q_eval, clf):
    # Evaluate the generative process using classical ML methods for classification
    x_train = torch.cat([p_samples, q_samples])
    y = torch.tensor([1.0] * len(p_samples) + [0.0] * len(q_samples), device=x_train.device)
    x_eval = torch.cat([p_eval, q_eval])
    y_eval = torch.tensor([1.0] * len(p_eval) + [0.0] * len(q_eval), device=x_eval.device)
    # shuffle data
    x_train, y = shuffle(x_train, y, random_state=0)
    # Data to cpu
    x_train = x_train.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    x_eval = x_eval.detach().cpu()
    y_eval = y_eval.detach().cpu()

    # clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=3, max_depth=2)
    clf.fit(x_train, y)
    y_pred = clf.predict(x_eval)
    acc = accuracy_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred)
    validation_predictions = clf.predict_proba(x_eval)
    validation_predictions = torch.tensor(validation_predictions[:, 1], dtype=torch.float32)
    # Calculate binary cross-entropy
    loss_i = torch.nn.BCELoss()
    if not isinstance(validation_predictions, torch.Tensor):
        validation_predictions = torch.tensor(validation_predictions, dtype=torch.float32)
    loss_j = loss_i(validation_predictions, y_eval)

    # training accuracy
    y_pred_train = clf.predict(x_train)
    acc_train = accuracy_score(y, y_pred_train)
    f1_train = f1_score(y, y_pred_train)
    print(f"Random forest Train: Accuracy: {acc_train}, F1: {f1_train}")
    return acc, f1, loss_j


def get_best_seed(cfg, results_path):
    # The generator model uses several seeds and hyperparameters so it is necessary to select the best one to
    # transfer knowledge.
    experiment_name = cfg.experiment_name
    dataset_name = results_path.split(experiment_name)[-1]
    dataset_name = dataset_name.split('_pretrained')[0]
    dataset_name = dataset_name.split(cfg.phase2.gen_model)[0]
    results_path = os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'results',
                                experiment_name + dataset_name + cfg.phase1.gen_model + '_phase2_' + cfg.phase2.gen_model)

    df_js = pd.read_csv(os.path.join(results_path, 'js.csv'))
    n = max(df_js['n'].unique())  # higher n means better generation
    m = max(df_js['m'].unique())  # higher m means better estimation of the divergence
    l = max(df_js['l'].unique())  # higher l means better estimation of the divergence
    disc_init_path = os.path.join(results_path, 'kl', f'{n}_{m}_{l}')

    df_jsi = df_js[(df_js['n'] == n) & (df_js['m'] == m) & (df_js['l'] == l)]
    best_seed_ph2 = df_jsi.iloc[df_js['JS Discriminator'].idxmin()]['name'].split('_')[-1]
    return best_seed_ph2, disc_init_path


def get_pretrained(cfg, results_path):
    # Get the path of the pretrained model
    if 'data' in cfg.case:
        best_seed_ph2, disc_init_path = get_best_seed(cfg, results_path)
        # Group by n, m, l and get the best seed
        # List of files in the folder
        files = os.listdir(disc_init_path)
        # Filter files with the desired extension
        if cfg.gen_model == 'vae':
            discriminators = [file for file in files if file.endswith('.pt') if f'_seed_{best_seed_ph2}' in file]
        elif cfg.gen_model == 'ctgan' or cfg.gen_model == 'tvae':
            discriminators = [file for file in files if file.endswith('.pt')]
        else:
            raise ValueError('Generator model not supported')

        pre_path = os.path.join(disc_init_path, discriminators[0])
        return pre_path
    elif 'mvn' in cfg.case:
        path = results_path.split('_pretrained')[0]
        df_js = pd.read_csv(os.path.join(path, 'js.csv')).reset_index()
        n = df_js['n'].max()
        m = df_js['m'].max()
        ls = df_js['l'].unique()
        # remove 5000 from ls
        ls = [l for l in ls if l != 5000]
        l = max(ls)

        pre_path = os.path.join(path, 'kl', f'{n}_{m}_{l}')
        files = os.listdir(pre_path)
        # Filter files with the desired extension
        discriminators = [file for file in files if file.endswith('.pt')]
        pre_path = os.path.join(pre_path, discriminators[0])

        return pre_path


class Divergence(ABC):

    def __init__(self, p_samples, q_samples, fraction_train, device='cpu', seed=0, pre_path=None,
                 results_path=None, n=0, m=0, l=0, layers=(256, 64, 32), dataset_name=None, cfg=None) -> None:

        self._model = Discriminator(layers)
        self._model.to(device=device)
        self.results_path = results_path
        self.seed = seed
        self.dataset_name = dataset_name
        # Load pre-trained model
        if pre_path is not None and os.path.exists(pre_path):
            # Check if file exists
            if os.path.isfile(pre_path):
                print('Loading pretrained model for the discriminator from cfg')
                self._model.load_state_dict(torch.load(pre_path))
        elif cfg is not None:
            if cfg.use_pretrained:
                print('Using pretrained model for the discriminator')
                pre_path = get_pretrained(cfg, results_path)
                self._model.load_state_dict(torch.load(pre_path))

        if len(p_samples) != len(q_samples):
            raise ValueError('p_samples and q_samples must have the same length')

        # Split data into train and eval
        if not 0.0 < fraction_train <= 1.0:
            raise ValueError()
        # Split data into train, validation (for early stopping) and test (to estimate)
        if fraction_train != 1.0:
            split_idx = int(len(p_samples) * fraction_train)
            self.p_train, self.p_eval = p_samples[:split_idx], p_samples[split_idx:]
            self.p_test, self.p_val = self.p_eval[:len(self.p_eval) // 2], self.p_eval[len(self.p_eval) // 2:]
            self.q_train, self.q_eval = q_samples[:split_idx], q_samples[split_idx:]
            self.q_test, self.q_val = self.q_eval[:len(self.q_eval) // 2], self.q_eval[len(self.q_eval) // 2:]
            real_m = len(self.p_train)
            real_l = len(self.p_val)
            if real_m != m or real_l != l:
                raise ValueError(f'Expected m={m} and l={l} but got m={real_m} and l={real_l}')

            self.n = n
            self.m = m
            self.l = l
        else:
            self.p_train, self.p_eval = p_samples, p_samples
            self.q_train, self.q_eval = q_samples, q_samples
            self.p_val = self.p_eval
            self.q_val = self.q_eval



    def to_dataloader(self, p_samples, q_samples=None, shuffle=True):
        # Convert tensors to dataloader
        if q_samples is not None:
            X = torch.cat([p_samples, q_samples])
            y = torch.tensor([1.0] * len(p_samples) + [0.0] * len(q_samples), device=X.device)
            return tensor_to_dataloader(X, y, shuffle=shuffle)

        else:
            return tensor_to_dataloader(p_samples, shuffle=shuffle)

    def evaluate_mlp(self, p_eval, q_eval):
        # Evaluate the generative process using classical metrics for classification
        # Evaluate mlp - validation set
        dl_2 = self.to_dataloader(p_eval, q_eval, shuffle=False)
        y_pred = self._model.predict(dl_2, sigmoid=True)
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred, dtype=torch.float32)
        # Compute the predicted class label for each example using where and if
        y_pred_label = torch.where(y_pred > 0.5, torch.tensor(1.0), torch.tensor(0.0))
        y_eval = dl_2.dataset.y
        acc_mlp = accuracy_score(y_eval.cpu(), y_pred_label.cpu())
        f1_mlp = f1_score(y_eval.cpu(), y_pred_label.cpu())
        loss_mlp = torch.nn.BCELoss()(y_pred, y_eval)
        return acc_mlp, f1_mlp, loss_mlp

    def fit(self, p_samples, q_samples, n_fit_epochs: int, p_eval, q_eval, n=0, cfg=None):
        # Fit the discriminator
        dl = self.to_dataloader(self.p_train, self.q_train)
        dl_eval = self.to_dataloader(self.p_val, self.q_val)
        # plt_tsne(self.p_train, self.q_train, self.results_path)
        df = pd.concat([pd.DataFrame(self.p_train.cpu().numpy()), pd.DataFrame(self.q_train.cpu().numpy())])
        df['target'] = [1] * len(self.p_train) + [0] * len(self.q_train)

        if isinstance(self._model, MLPClassifier):
            self._model.fit(dl.dataset.X, dl.dataset.y)
        else:
            self._model.train_loop(dl, n_fit_epochs, dl_eval=dl_eval, n=self.n, seed=self.seed, cfg=cfg)

        # Evaluate the generative process using classical ML methods for classification
        acc_rf, f1_rf, loss = ml_comparison(self.p_train, self.q_train, self.p_val, self.q_val,
                                            RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=3,
                                                                   max_depth=2))
        acc_log, f1_log, _ = ml_comparison(self.p_train, self.q_train, self.p_val, self.q_val,
                                           LogisticRegression(random_state=0))

        # Evaluate mlp - validation set
        acc_mlp, f1_mlp, loss_mlp = self.evaluate_mlp(self.p_eval, self.q_eval)
        acc_mlp_train, f1_mlp_train, loss_mlp_train = self.evaluate_mlp(self.p_train, self.q_train)
        acc_mlp_test, f1_mlp_test, loss_mlp_test = self.evaluate_mlp(self.p_test, self.q_test)

        # # Print results
        print(f"Logistic Regression: Accuracy: {acc_log}, F1: {f1_log}")
        print(f"Random Forest: Accuracy: {acc_rf}, F1: {f1_rf}, Loss: {loss}")
        print(f"MLP Validation: Accuracy: {acc_mlp}, F1: {f1_mlp}, Loss: {loss_mlp}")
        print(f"MLP Train: Accuracy: {acc_mlp_train}, F1: {f1_mlp_train}, Loss: {loss_mlp_train}")
        print(f"MLP Test: Accuracy: {acc_mlp_test}, F1: {f1_mlp_test}, Loss: {loss_mlp_test}")

    @abstractmethod
    def estimate(self, p_samples, q_samples) -> torch.Tensor:
        ...

    def forward(self, *, n_fit_epochs: int, n=0, save_model=False,
                path='model.pt', cfg=None):

        self.fit(self.p_train, self.q_train, n_fit_epochs, self.p_val, self.q_val, n, cfg=cfg)
        estimate = self.estimate(self.p_test, self.q_test)

        if save_model:
            torch.save(self._model.state_dict(), path)

        return estimate

    def pr_sample_probs(self, pr, x_real):
        ratio = 0
        for i, _ in enumerate(pr.mixture_distribution.probs):
            pi = pr.mixture_distribution.probs[i]
            normal = D.MultivariateNormal(pr.component_distribution.loc[i],
                                          pr.component_distribution.covariance_matrix[i])
            ratio += pi * torch.exp(normal.log_prob(x_real))
        return ratio

    def qs_sample_probs(self, qs, x_gen):
        ratio = 0
        for i, _ in enumerate(qs.gmm.weights_):
            pi_hat = qs.gmm.weights_[i]
            normal_hat = D.MultivariateNormal(torch.tensor(qs.gmm.means_[i]), torch.tensor(qs.gmm.covariances_[i]))
            ratio += pi_hat * torch.exp(normal_hat.log_prob(x_gen))
        return ratio


class KL(Divergence):

    def mc(self, pr, qs, p_test=None):
        if p_test is None:
            p_test = self.p_test

        # check device, if cuda, move to cpu
        if p_test.device.type == 'cuda':
            p_test = p_test.cpu().detach()
        log_real = pr.log_prob(p_test)
        log_fake = qs.log_prob(p_test)

        return (log_real - log_fake).mean()

    def perfect_ratio(self, pr, qs, p_test=None):
        if p_test is None:
            p_test = self.p_test

        # check device, if cuda, move to cpu
        if p_test.device.type == 'cuda':
            p_test = p_test.cpu().detach()

        num = self.pr_sample_probs(pr, p_test)
        den = self.qs_sample_probs(qs, p_test)
        r = num / den
        return torch.log(r).mean()

    def estimate(self, p_samples, *_):
        dl = tensor_to_dataloader(self.p_test)
        if isinstance(self._model, MLPClassifier):
            probs = self._model.predict_proba(dl.dataset.X)[:, 1]
            log_ratio = torch.log(torch.tensor(probs) / (1 - torch.tensor(probs)))
        else:
            log_ratio = self._model.predict(dl, sigmoid=False)

        estimate = torch.mean(log_ratio)
        return estimate


class JS(Divergence):

    def mc(self, pr, qs, x_real=None, x_gen=None):
        if x_real is None or x_gen is None:
            x_real = self.p_test
            x_gen = self.q_test

        # check device, if cuda, move to cpu
        if x_real.device.type == 'cuda':
            x_real = x_real.cpu().detach()
            x_gen = x_gen.cpu().detach()

        p_real = torch.exp(pr.log_prob(x_real))
        q_real = torch.exp(qs.log_prob(x_real))

        p_gen = torch.exp(pr.log_prob(x_gen))
        q_gen = torch.exp(qs.log_prob(x_gen))

        t1 = torch.log2(2 * p_real) - torch.log2(q_real + p_real)
        t2 = (torch.log2(2 * q_gen) - torch.log2(p_gen + q_gen))

        return (t1.mean() + t2.mean()) / 2

    def perfect_ratio(self, pr, qs, x_real, x_gen):
        # check device, if cuda, move to cpu
        if x_real.device.type == 'cuda':
            x_real = x_real.cpu().detach()
            x_gen = x_gen.cpu().detach()
        p_real = self.pr_sample_probs(pr, x_real)
        q_real = self.qs_sample_probs(qs, x_real)

        p_gen = self.pr_sample_probs(pr, x_gen)
        q_gen = self.qs_sample_probs(qs, x_gen)

        di_real_log = torch.log2(p_real) - torch.log2(p_real + q_real)
        di_gen = torch.exp(torch.log(p_gen) - torch.log(p_gen + q_gen))
        di_gen_log = torch.log2(1 - di_gen)

        t1 = (di_real_log.mean()) / 2
        t2 = (di_gen_log.mean()) / 2
        js = t1 + t2 + torch.log2(torch.tensor([2.]))
        return js

    def js_and_bound(self, p, q):
        # Estimate the js divergence and the bound to the loss function
        p_dl = tensor_to_dataloader(p)
        q_dl = tensor_to_dataloader(q)
        if isinstance(self._model, MLPClassifier):
            prob_p = self._model.predict_proba(p_dl.dataset.X)[:, 1]
            prob_q = 1 - self._model.predict_proba(q_dl.dataset.X)[:, 1]
            if not isinstance(prob_p, torch.Tensor):
                prob_p = torch.tensor(prob_p)
                prob_q = torch.tensor(prob_q)
        else:
            prob_p = self._model.predict(p_dl, sigmoid=True)
            prob_q = 1 - self._model.predict(q_dl, sigmoid=True)

        # ax = self._model.loss_plot
        prob_p = torch.clamp(prob_p, min=1e-7, max=1)
        prob_q = torch.clamp(prob_q, min=1e-7, max=1)
        estimate = 0.5 * (1 + torch.log2(prob_p).mean() + 1 + torch.log2(prob_q).mean())

        estimate_ln = 0.5 * ((torch.log(prob_p)).mean()) + 0.5 * ((torch.log((prob_q))).mean()) + torch.log(
            torch.tensor(2))

        # print(f"JS estimate: {estimate}, {estimate_ln}")
        # plot horizontal line
        val_bound = (-2 * estimate_ln) + torch.log(torch.tensor(4))

        return estimate, val_bound

    def estimate(self, p_samples, q_samples):
        # Estimate the js divergence for the test set
        estimate, val_bound = self.js_and_bound(self.p_test, self.q_test)

        # Estimate the js divergence for the train set
        estimate_train, train_bound = self.js_and_bound(self.p_train, self.q_train)
        print(f"JS estimate: {estimate}, JS estimate train: {estimate_train}")

        # Plot loss function and the bounds of the js divergence
        ax = self._model.loss_plot

        ax.axhline(val_bound.cpu(), color='r')
        ax.axhline(train_bound.cpu(), color='g')

        plt.title('Loss function for JS')
        if self.results_path is not None:
            fig_path = os.path.join(self.results_path, 'js', f'{self.n}_{self.m}_{self.l}',
                                    f'seed_{self.seed}_js_loss_{self.dataset_name}.')
            plt.savefig(fig_path + 'png')
            # tikzplotlib.save(fig_path + 'tex')
            # plt.savefig(os.path.join(self.results_path, 'js', f'{self.n}_{self.m}_{self.l}',
            #                          f'seed_{self.seed}_js_loss_{self.dataset_name}.png'))
        else:
            plt.savefig("js_loss.png")
            # tikzplotlib.save("js_loss.tex")

        plt.show()

        return estimate
