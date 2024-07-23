from sklearn.mixture import GaussianMixture
import torch

class GMM():

    def __init__(self, n_components=2, covariance_type='full', random_state=0):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.gmm = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,
                                   random_state=self.random_state)

    def fit(self, X):
        self.gmm.fit(X)
        return self.gmm

    def sample(self, n_samples):
        return self.gmm.sample(n_samples)

    def log_prob(self, X):
        device = X.device
        return torch.tensor(self.gmm.score_samples(X), device=device)