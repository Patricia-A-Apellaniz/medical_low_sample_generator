import os
import sys
import time
import copy
import torch

from pathlib import Path
import matplotlib.pyplot as plt

from torch import nn, optim
from .dense import DenseModule
from torch.nn import functional as F
from torch.utils.data import DataLoader


def plot_losses(losses, losses_eval, dataloader, dl_eval, n, name, seed=0, learning_rates=None):
    loss_path = os.path.join(os.path.join(Path(sys.argv[0]).resolve().parent.parent, 'results', 'losses'))
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)
    fig, ax = plt.subplots()
    losses = losses
    losses_eval = losses_eval
    ax.plot(losses)
    if dl_eval is not None:
        plt.plot(losses_eval)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f'Discriminator Loss (M={dataloader.dataset.X.shape[0] / 2}, L={dl_eval.dataset.X.shape[0] / 2}, N={n}) for {name}')
    return ax


class Discriminator(nn.Module):
    def __init__(self, layers,*args, **kwargs):
        super().__init__(*args, **kwargs)
        layers_ = []
        for elem in layers:
            layers_.append(DenseModule(elem, activation="leaky_relu", batch_norm=True, dropout=True))
        layers_ += [nn.LazyLinear(1)]
        self.l = nn.ModuleList(layers_)
        self.loss_plot = None

    def forward(self, data: torch.Tensor, *, sigmoid=False) -> torch.Tensor:
        x = data
        for layer in self.l:
            x = layer(x)
        if sigmoid:
            x = F.sigmoid(x)
        return x.reshape(-1)

    def train_loop(self, dataloader: DataLoader, n_epoch: int, optimizer=None, dl_eval=None, n=0, seed=0, cfg=None):
        if optimizer is None:
            if cfg is not None and hasattr(cfg, 'lr'):
                #print(f'Using learning rate {cfg.lr}')
                optimizer = optim.Adam(self.parameters(), lr=cfg.lr)
            else:
                optimizer = optim.Adam(self.parameters(), 1e-3)
        self.train(True)
        losses = []
        losses_eval = []
        # Set up early stopping parameters
        best_metric = float('inf')  # For loss, set it to float('inf'); for accuracy, set it to 0
        patience_0 = 1000  # Number of epochs to wait before stopping
        patience = patience_0  # Number of epochs to wait before stopping
        best_model = None

        t_0 = time.time()
        learning_rates = []

        if len(dataloader) == 1: # Only a single batch of data
            X, y = next(iter(dataloader))  # Already get the data, this weirdly goes faster than using the dataloader

        for epoch in (range(n_epoch)):
            if (epoch + 1) % 500 == 0:
                print(f"Discriminator estimator: Epoch [{epoch+1}/{n_epoch}], Time since start: {time.time() - t_0} seconds (average: {(time.time() - t_0) / (epoch+1)} seconds per epoch)")
            cum_loss = 0.0
            cum_loss_eval = 0.0
            self.train(True)
            if len(dataloader) == 1:  # Only a single batch of data
                optimizer.zero_grad()
                logit_X = self(X)
                loss = F.binary_cross_entropy_with_logits(logit_X, y.reshape(-1))
                loss.backward()
                optimizer.step()
                cum_loss += loss.item()
            else:
                for X, y in dataloader:
                    optimizer.zero_grad()
                    logit_X = self(X)
                    loss = F.binary_cross_entropy_with_logits(logit_X, y.reshape(-1))
                    loss.backward()
                    optimizer.step()
                    cum_loss += loss.item()
            avg_loss = cum_loss / len(dataloader)
            losses.append(avg_loss*2)
            # For early stopping
            if dl_eval is not None:
                self.eval()
                with torch.no_grad():
                    for X_eval, y_eval in dl_eval:
                        logit_X_eval = self(X_eval)
                        loss_eval = F.binary_cross_entropy_with_logits(logit_X_eval, y_eval.reshape(-1))
                        cum_loss_eval += loss_eval.item()

                    avg_loss_eval = cum_loss_eval / (len(dl_eval))
                    losses_eval.append(avg_loss_eval*2)
                # Early stopping
                # Check if the validation loss (or accuracy) has improved
                if avg_loss_eval < best_metric:
                    best_metric = avg_loss_eval
                    patience = patience_0  # Reset patience
                    best_model = copy.deepcopy(self.state_dict())
                else:
                    patience -= 1

                if patience == 0:
                    print("Early stopping, no improvement in validation loss.")
                    self.load_state_dict(best_model)
                    break
        self.loss_plot = plot_losses(losses, losses_eval, dataloader, dl_eval, n, name=cfg.name, seed=seed, learning_rates=learning_rates)

    @torch.no_grad()
    def predict(self, data: DataLoader, *, sigmoid=True):
        self.train(False)
        _X_y = 2
        out = []
        for batch in data:
            if len(batch) == _X_y:
                x, _ = batch
            else:
                x = batch
            y = self.forward(x, sigmoid=sigmoid)
            out.append(y)
        return torch.cat(out)

