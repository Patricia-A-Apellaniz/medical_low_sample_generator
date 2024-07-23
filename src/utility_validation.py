import os
import time
import copy
import torch

import numpy as np
import pandas as pd

from torch import nn, optim
from gen_model.savae import SAVAE
from torch.nn import functional as F
from sklearn.model_selection import KFold
from divergence_estimation.dense import DenseModule
from sklearn.metrics import f1_score, accuracy_score



## UTILITY VALIDATION FOR CLASSIFICATION
class Classifier(nn.Module):  # Classifier, based on the one used to approximate the divergences
    def __init__(self, n_out, layers=(256, 64, 32), *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers_ = []
        for elem in layers:
            layers_.append(DenseModule(elem, activation="leaky_relu", batch_norm=True, dropout=True))
        layers_ += [nn.LazyLinear(n_out)]
        self.l = nn.ModuleList(layers_)
        self.n_out = n_out

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x = data
        for layer in self.l:
            x = layer(x)
        return x # Return the logits

    def fit(self, X:torch.Tensor, y:torch.Tensor, n_epoch: int, optimizer=None, X_eval=None, y_eval=None, cfg=None):

        if optimizer is None:
            if cfg is not None and hasattr(cfg, 'lr'):
                optimizer = optim.Adam(self.parameters(), lr=cfg.lr)
            else:
                optimizer = optim.Adam(self.parameters(), 1e-3)  # Default learning rate

        # Compute weights to account for class imbalance (as medical datasets tend to have a lot of imbalance)
        n_samples = len(y)
        class_counts = torch.bincount(y, minlength=self.n_out)
        weights = n_samples / (self.n_out * class_counts)

        self.train(True)
        losses = []
        losses_eval = []

        # Set up early stopping parameters
        best_metric = float('inf')  # For loss, set it to float('inf'); for accuracy, set it to 0
        patience_0 = 50  # Number of epochs to wait before stopping
        patience = patience_0  # Number of epochs to wait before stopping
        best_model = None

        t_0 = time.time()

        for epoch in (range(n_epoch)):
            if (epoch + 1) % 100 == 0:
                print(f"Utility classifier: Epoch [{epoch+1}/{n_epoch}], Time since start: {time.time() - t_0} seconds (average: {(time.time() - t_0) / (epoch+1)} seconds per epoch)")
            self.train(True)
            optimizer.zero_grad()
            logit_X = self(X)
            loss = F.cross_entropy(logit_X, y, weight=weights)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # For early stopping
            if X_eval is not None:
                self.eval()
                with torch.no_grad():
                    logit_X_eval = self(X_eval)
                    loss_eval = F.cross_entropy(logit_X_eval, y_eval, weight=weights)
                    losses_eval.append(loss_eval.item())
                # Early stopping
                if epoch == 0:
                    best_model = copy.deepcopy(self.state_dict())  # Save the initial model, we'll overwrite it later
                # Check if the validation loss (or accuracy) has improved
                if loss_eval.item() < best_metric:
                    best_metric = loss_eval.item()
                    patience = patience_0  # Reset patience
                    best_model = copy.deepcopy(self.state_dict())
                else:
                    patience -= 1

                if patience == 0:
                    print(f"Early stopping at iteration {epoch + 1}, no improvement in validation loss.")
                    self.load_state_dict(best_model)
                    break

    @torch.no_grad()
    def predict(self, X_pred: torch.Tensor):
        self.train(False)
        y = self.forward(X_pred)
        return torch.argmax(y, dim=1).numpy()


def evaluate_classification_metrics(args): # TODO: this could be parallelized and sent to GPU, although it is already quite fast
    os.makedirs(args['output_dir'], exist_ok=True)
    syn_df = args['syn_df']
    real_df = args['real_df']
    util_val_df = args['util_val_df']
    n_classes = max((len(np.unique(real_df.iloc[:, -1])), len(np.unique(syn_df.iloc[:, -1])), len(np.unique(util_val_df.iloc[:, -1]))))
    if n_classes > 2:  # If the problem is not binary, then we cannot use F1
        problem_type = 'multiclass'
    else:
        problem_type = 'binary'
    print(f"Utility validation for {args['dataset_name']} dataset, using classification: {problem_type} problem.")
    results = [{'f1': [], 'acc': []} for _ in range(3)]

    # Use k-fold on real data to get a better estimate of the performance
    kf = KFold(n_splits=args['n_splits'], shuffle=True, random_state=0)
    for train_index, test_index in kf.split(real_df):
        real_train = real_df.iloc[train_index, :]
        real_test = util_val_df  # NOTE: validate always with the same data!!

        # Case 1: Train a classifier on the real data and evaluate on the real test data
        clf = Classifier(n_classes)
        clf.fit(torch.from_numpy(real_train.iloc[:, :-1].values).float(),
                torch.from_numpy(real_train.iloc[:, -1].values).long(),
                n_epoch=10000,
                X_eval=torch.from_numpy(real_test.iloc[:, :-1].values).float(),
                y_eval=torch.from_numpy(real_test.iloc[:, -1].values).long())
        case_1_pred = clf.predict(torch.from_numpy(real_test.iloc[:, :-1].values).float())

        if problem_type == 'binary':
            case_1_f1 = f1_score(real_test.iloc[:, -1], case_1_pred)
        else:
            case_1_f1 = -1
        case_1_acc = accuracy_score(real_test.iloc[:, -1], case_1_pred)
        results[0]['f1'].append(case_1_f1)
        results[0]['acc'].append(case_1_acc)

        # Case 2: Train a classifier on the synth data and test on the real data
        clf = Classifier(n_classes)
        clf.fit(torch.from_numpy(syn_df.iloc[:, :-1].values).float(),
                torch.from_numpy(syn_df.iloc[:, -1].values).long(),
                n_epoch=10000,
                X_eval=torch.from_numpy(real_test.iloc[:, :-1].values).float(),
                y_eval=torch.from_numpy(real_test.iloc[:, -1].values).long())
        case_2_pred = clf.predict(torch.from_numpy(real_test.iloc[:, :-1].values).float())
        if problem_type == 'binary':
            case_2_f1 = f1_score(real_test.iloc[:, -1], case_2_pred)
        else:
            case_2_f1 = -1
        case_2_acc = accuracy_score(real_test.iloc[:, -1], case_2_pred)
        results[1]['f1'].append(case_2_f1)
        results[1]['acc'].append(case_2_acc)

        # Case 3: fine tune the previous classifier using real data
        clf.fit(torch.from_numpy(real_train.iloc[:, :-1].values).float(),
                torch.from_numpy(real_train.iloc[:, -1].values).long(),
                n_epoch=10000,
                X_eval=torch.from_numpy(real_test.iloc[:, :-1].values).float(),
                y_eval=torch.from_numpy(real_test.iloc[:, -1].values).long())
        case_3_pred = clf.predict(torch.from_numpy(real_test.iloc[:, :-1].values).float())
        if problem_type == 'binary':
            case_3_f1 = f1_score(real_test.iloc[:, -1], case_3_pred)
        else:
            case_3_f1 = -1
        case_3_acc = accuracy_score(real_test.iloc[:, -1], case_3_pred)
        results[2]['f1'].append(case_3_f1)
        results[2]['acc'].append(case_3_acc)

    # Save the results as a CSV
    df = pd.DataFrame({'case': ['real', 'synth', 'fine_tune'], 'f1': [np.mean(results[0]['f1']), np.mean(results[1]['f1']), np.mean(results[2]['f1'])],
                       'acc': [np.mean(results[0]['acc']), np.mean(results[1]['acc']), np.mean(results[2]['acc'])],
                       'f1_std': [np.std(results[0]['f1']), np.std(results[1]['f1']), np.std(results[2]['f1'])],
                       'acc_std': [np.std(results[0]['acc']), np.std(results[1]['acc']), np.std(results[2]['acc'])]})

    df.to_csv(os.path.join(args['output_dir'], 'classification_metrics.csv'), index=False)


## UTILITY VALIDATION FOR SURVIVAL ANALYSIS
def evaluate_survival_metrics(args):
    os.makedirs(args['output_dir'], exist_ok=True)
    syn_df = args['syn_df']
    real_df = args['real_df']
    util_val_df = args['util_val_df']
    print(f"Utility validation for {args['dataset_name']} dataset, using survival analysis.")
    results = [{'ci': [], 'ibs': []} for _ in range(3)]

    # Since the time is generated, it may be negative (i.e., a gaussian was used for time generation). In SAVAE, time is modelled as a Weibull (always positive), so ensure that time is strictly positive
    min_t = min([np.amin(real_df['time']), np.amin(syn_df['time']), np.amin(util_val_df['time'])])
    syn_df['time'] = syn_df['time'] - (min_t - 0.01)
    real_df['time'] = real_df['time'] - (min_t - 0.01)
    util_val_df['time'] = util_val_df['time'] - (min_t - 0.01)

    # SAVAE hiperparameters
    max_t = max([np.amax(real_df['time']), np.amax(syn_df['time'])])

    feat_distributions = copy.deepcopy(args['metadata']['feat_distributions'][0:-2])  # For survival, keep only covariates distributions
    assert real_df.columns[-2] == 'time' and real_df.columns[-1] == 'event' # To ensure that the columns are properly ordered

    model_params = {'feat_distributions': feat_distributions,
                    'latent_dim': 10,
                    'hidden_size': 256,
                    'input_dim': len(feat_distributions),
                    'max_t': max_t,
                    'time_dist': ('weibull', 2),
                    'early_stop': True}
    train_params = {'n_epochs': 5000, 'batch_size': 256, 'device': torch.device('cpu'), 'lr': 1e-3, 'path_name': None}

    kf = KFold(n_splits=args['n_splits'], shuffle=True, random_state=0)
    for train_index, test_index in kf.split(real_df):
        real_train = real_df.iloc[train_index, :]
        real_test = util_val_df  # NOTE: validate always witht the same data!!
        censor_val = real_test['event'].values  # Censoring for validation

        # Case 1: train and validate on real data
        data = (real_train, pd.DataFrame(np.ones_like(real_train), columns=real_train.columns),
                real_test, pd.DataFrame(np.ones_like(real_test), columns=real_test.columns))
        time_train = real_train['time'].values
        model = SAVAE(model_params)
        _ = model.fit(data, train_params)
        ci, ibs = model.calculate_risk(time_train, real_test, censor_val)
        results[0]['ci'].append(ci)
        results[0]['ibs'].append(ibs)

        # Case 2: train on synthetic data and validate on real data
        data = (syn_df, pd.DataFrame(np.ones_like(syn_df), columns=syn_df.columns),
                real_test, pd.DataFrame(np.ones_like(real_test), columns=real_test.columns))
        time_train = syn_df['time'].values
        model = SAVAE(model_params)
        _ = model.fit(data, train_params)
        ci, ibs = model.calculate_risk(time_train, real_test, censor_val)
        results[1]['ci'].append(ci)
        results[1]['ibs'].append(ibs)

        # Case 3: train on synthetic data and fine-tune on real data
        data = (real_train, pd.DataFrame(np.ones_like(real_train), columns=real_train.columns),
                real_test, pd.DataFrame(np.ones_like(real_test), columns=real_test.columns))
        time_train = real_train['time'].values
        _ = model.fit(data, train_params)  # Fine tune the previous model (already trained on synthetic data)
        ci, ibs = model.calculate_risk(time_train, real_test, censor_val)
        results[2]['ci'].append(ci)
        results[2]['ibs'].append(ibs)

    # Save the results as a CSV
    df = pd.DataFrame({'case': ['real', 'synth', 'fine_tune'], 'ci': [np.mean(results[0]['ci']), np.mean(results[1]['ci']), np.mean(results[2]['ci'])],
                       'ibs': [np.mean(results[0]['ibs']), np.mean(results[1]['ibs']), np.mean(results[2]['ibs'])],
                       'ci_std': [np.std(results[0]['ci']), np.std(results[1]['ci']), np.std(results[2]['ci'])],
                       'ibs_std': [np.std(results[0]['ibs']), np.std(results[1]['ibs']), np.std(results[2]['ibs'])]})

    df.to_csv(os.path.join(args['output_dir'], 'survival_metrics.csv'), index=False)


