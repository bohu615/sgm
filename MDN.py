import torch
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical

class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network.

    [ Bishop, 1994 ]

    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components):
        super().__init__()
        self.pi_network = CategoricalNetwork(dim_in, n_components)
        self.normal_network = MixtureDiagNormalNetwork(dim_in, dim_out,
                                                       n_components)

    def forward(self, x):
        return self.pi_network(x), self.normal_network(x)

    def loss(self, x, y):
        pi, normal = self.forward(x)
        loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
        loglik = torch.sum(loglik, dim=2)
        loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=1)
        return loss

    def sample(self, x):
        pi, normal = self.forward(x)
        samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)
        return samples

class MixtureDiagNormalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, n_components, hidden_dim=None):
        super().__init__()
        self.n_components = n_components
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * out_dim * n_components),
        )

    def forward(self, x):
        params = self.network(x)
        mean, sd = torch.split(params, params.shape[1] // 2, dim=1)
        mean = torch.stack(mean.split(mean.shape[1] // self.n_components, 1))
        sd = torch.stack(sd.split(sd.shape[1] // self.n_components, 1))
        return Normal(mean.transpose(0, 1), torch.exp(sd).transpose(0, 1))

class CategoricalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        params = self.network(x)
        return OneHotCategorical(logits=params)
    
def construct_contour1d(centers, weights, learned_variance, interp=100):
    QMI_TRUE_LIST = []
    min = 0
    max = 1
    delta = (max-min)/interp

    x_axis = np.linspace(min, max, interp)

    gaussian_plot_joint_ = []
    gaussian_plot_split_x_ = []
    gaussian_plot_split_y_ = []

    gaussian_plot_joint_ = (weights*gaussian_1d(x_axis, centers, learned_variance))/np.sum(weights)
    gaussian_plot_joint = np.mean(np.array(gaussian_plot_joint_), 1)/delta
    
    return gaussian_plot_joint

def visualize_mdn():
    interp = 200
    scanning_input = torch.from_numpy(np.linspace(0, 1, interp)).float().cuda().reshape(-1, 1)
    pi, normal = model.forward(scanning_input)
    
    means = normal.loc.detach().cpu().numpy()
    vars = normal.scale.detach().cpu().numpy()
    pis = pi.probs.detach().cpu().numpy()
    
    gaussian_plot_joint_ = []
    for i in range(0, interp):
        gaussian_plot_joint = construct_contour1d(means[i, :].reshape(-1), pis[i, :].reshape(-1), vars[i, :].reshape(-1)**2, interp=100)
        #plt.plot(np.linspace(0, 1, 100), gaussian_plot_joint)
        gaussian_plot_joint_.append(gaussian_plot_joint)

    plt.rcParams["figure.figsize"] = [4,4]
    model_pdf = np.array(gaussian_plot_joint_)/np.sum(np.array(gaussian_plot_joint_), 1).reshape(-1, 1)

#     plt.imshow(normalize, origin='lower', extent=[min, max, min, max])
#     plt.show()

    plt.imshow(model_pdf, origin='lower', extent=[min, max, min, max])
    plt.show()
    
x = current_[:].reshape(-1)
y = next_[:].reshape(-1)

from argparse import ArgumentParser
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

def gen_data(n=512):
    y = np.linspace(-1, 1, n)
    x = 7 * np.sin(5 * y) + 0.5 * y + 0.5 * np.random.randn(*y.shape)
    return x[:,np.newaxis], y[:,np.newaxis]

def plot_data(x, y):
    plt.hist2d(x, y, bins=35)
    plt.xlim(-8, 8)
    plt.ylim(-1, 1)
    plt.axis('off')


x = current_[:].reshape(-1, 1)
y = next_[:].reshape(-1, 1)
x = torch.Tensor(x)
y = torch.Tensor(y)

model = MixtureDensityNetwork(1, 1, n_components=300).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
bs = 1000
 
for i in range(50000):
    optimizer.zero_grad()
    
    b1 = np.random.choice(x.shape[0], bs)
    x_i = (x[b1].reshape(-1, 1)).float().cuda()
    y_i = (y[b1].reshape(-1, 1)).float().cuda()
    
    loss = model.loss(x_i, y_i).mean()
    loss.backward()
    optimizer.step()
    if i % 1000 == 0:
        print(f"Iteration: {i}\t, " + f"Loss: {loss.data:.2f}")
        
        plt.title('Iteration:{0} - model conditional'.format(i))
        visualize_mdn()