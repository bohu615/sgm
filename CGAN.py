import numpy as np
import matplotlib.pyplot as plt

def construct_contour_gauss(centers, weights, learned_variance, interp=100):
    QMI_TRUE_LIST = []
    interp = 100
    delta = (max-min)/interp

    x_axis = np.linspace(min, max, interp)
    y_axis = np.linspace(min, max, interp)
    xv, yv = np.meshgrid(x_axis,y_axis)

    input = np.array((xv, yv)).reshape(2, -1).T

    gaussian_plot_joint_ = []
    gaussian_plot_split_x_ = []
    gaussian_plot_split_y_ = []

    #centers = np.concatenate((center_x, center_y), 1)
    #difference = input.reshape(-1, 2) - centers.reshape(-1, 1, 2)

    for i in range(0, centers.shape[0]):
        gaussian_plot_joint_.append(weights[i]*gaussian_nd(input - centers[i], 0, learned_variance[i]))
    gaussian_plot_joint = np.mean(np.array(gaussian_plot_joint_), 0)*delta*delta
    
    return gaussian_plot_joint.reshape(interp, interp)

def gaussian_nd_numpy(MEAN, VARIANCE):
    bs = VARIANCE.shape[0]
    dim = VARIANCE.shape[1]

    det = np.linalg.det(VARIANCE)
    inv = np.linalg.pinv(VARIANCE)
            
    product = np.sum((MEAN.reshape(-1, 1, dim)@inv).reshape(-1, dim)*MEAN.reshape(-1, dim), 1)
    
    return ((2*np.pi)**(-dim/2))*det**(-1/2)*np.exp(-(1/2)*product)

def compute_TRUE_ENTROPY(MEAN_matrix, COV_matrix, weights_matrix):
    K = MEAN_matrix.shape[0]
    dim = MEAN_matrix.shape[1]
    
    MEAN_DIFF = MEAN_matrix.reshape(K, 1, dim) - MEAN_matrix.reshape(1, K, dim)
    COV_DIFF = COV_matrix.reshape(K, 1, dim, dim) + COV_matrix.reshape(1, K, dim, dim)
    WEIGHT_DIFF = weights_matrix.reshape(K, 1)*weights_matrix.reshape(1, K)
    
    return np.sqrt(np.sum(WEIGHT_DIFF.reshape(-1)*gaussian_nd_numpy(MEAN_DIFF.reshape(-1, dim), COV_DIFF.reshape(-1, dim, dim))))

def gaussian_nd(input, m, sigma):
    k = sigma.shape[0]
    det = np.linalg.det(sigma)
    inv = np.linalg.pinv(sigma)
    
    return ((2*np.pi)**(-k/2))*det**(-1/2)*np.exp(-(1/2)*np.sum((input-m)@inv*(input-m), 1))

#### CONSTRUCT A MORE RESONABLE MKM

min = 0
max = 1

def makediag3d(a):
    a = np.asarray(a)
    depth, size = a.shape
    x = np.zeros((depth,size,size))
    for i in range(depth):
        x[i].flat[slice(0,None,1+size)] = a[i]
    return x

def create_Gaussian_mixture_MC():
    np.random.seed(4)
    
    num = 10
    center_x = np.linspace(0.2, 0.8, num)
    center_y = np.linspace(0.2, 0.8, num)
    
    np.random.shuffle(center_x)
    np.random.shuffle(center_y)

    xv, yv = np.meshgrid(center_x, center_y)

    MEAN_MATRIX = np.array((xv, yv)).reshape(2, -1).T
    COV = makediag3d(np.random.uniform(0.0005, 0.002, MEAN_MATRIX.shape[0]*2).reshape(num*num, 2))        
    weights = np.ones((num, num))*(0.3/(num-1))+np.eye(num)*(0.7-0.3/(num-1))
    
    return MEAN_MATRIX, COV, weights, center_x, center_y

MEAN_matrix, COV_matrix, weights_matrix, center_x, center_y = create_Gaussian_mixture_MC()
COV_matrix = COV_matrix*3
weights_matrix = weights_matrix.reshape(-1)

pdf = construct_contour_gauss(MEAN_matrix, weights_matrix.reshape(-1), COV_matrix)
normalize = pdf/np.sum(pdf,1).reshape(-1, 1)

def gaussian_1d(input, m, sigma):
    det = sigma
    inv = 1/sigma
        
    input = input.reshape(-1, 1)
    m = m.reshape(1, -1)

    return ((2*np.pi)**(-1/2))*det**(-1/2)*np.exp(-(1/2)*((input-m)**2*inv))

def generate_gauss_samples_various(MEAN, COV_matrix, samples_per_class=3000000):
    
    num_class = MEAN.shape[0]
    component_ = []
    for i in range(0, num_class):
        COV = COV_matrix[i]
        samples = np.random.normal(MEAN[i], np.sqrt(COV), int(samples_per_class))
        component_.append(samples)
    component_ = np.array(component_)
    
    return component_

np.random.seed(4)

inter = 100

x_0 = np.linspace(0.1, 0.9, inter)
#x_0 = np.array([0.5])
next_samples = np.copy(x_0)
density = []
stored_samples = generate_gauss_samples_various(MEAN_matrix[:, 0], COV_matrix[:, 0, 0])

iter = 1000

current_ = np.zeros((iter, inter))
next_ = np.zeros((iter, inter))
num_samples = np.zeros((weights_matrix.shape[0]), dtype=int)

print('generating dataset...')

for i in range(0, iter):
    current_[i] = np.copy(next_samples)
    weights_x0 = weights_matrix*gaussian_1d(next_samples, MEAN_matrix[:, 1], (COV_matrix[:, 1, 1]))
    weights_x0 = weights_x0/np.sum(weights_x0, 1).reshape(-1, 1)

    next_chosen = np.array([np.random.choice(weights_.shape[0], 1, p=weights_)[0] for weights_ in weights_x0])
    next_samples = []
    for j in next_chosen:
        next_samples.append(stored_samples[j, num_samples[j]])
        num_samples[j]+=1
    next_samples = np.array(next_samples)
    density.append(np.histogram(next_samples, bins=100)[0])
    
    next_[i] = np.copy(next_samples)
    
print('done!')

x = current_[:].reshape(-1)
y = next_[:].reshape(-1)

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

opt = {}

n_epochs= 200
batch_size = 64
lr = 0.00001
b1 = 0.5
b2 = 0.999
n_cpu = 1
latent_dim = 100
n_classes = 1
img_size = 32
channels = 1
sample_interval = 400
img_shape = (1, 1)

rand = 1
d_class = 10

def sample_discrete_class(bs=3000, how_many=5, d_class=5):
  return torch.nn.functional.one_hot(torch.randint(d_class, (bs*how_many,)), num_classes=d_class).view(bs, -1).float()


cuda = True


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(rand + d_class + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat(((labels).reshape(-1, 1), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), (labels).reshape(-1, 1)), -1)
        validity = self.model(d_in)
        return validity


# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
#os.makedirs("./mnist", exist_ok=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

x = current_[:].reshape(-1)
y = next_[:].reshape(-1)

# ----------
#  Training
# ----------

print('Now Running CGAN...')

batch_size = 64
bs = batch_size

for j in range(0, 100000):

    b1 = np.random.choice(x.shape[0], bs)
    x_i = torch.from_numpy(x[b1].reshape(-1, 1)).float().cuda()
    y_i = torch.from_numpy(y[b1].reshape(-1, 1)).float().cuda()

    b1 = np.random.choice(x.shape[0], bs)
    x_i_2 = torch.from_numpy(x[b1].reshape(-1, 1)).float().cuda()
    y_i_2 = torch.from_numpy(y[b1].reshape(-1, 1)).float().cuda()

    # Adversarial ground truths
    valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
    fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

    # Configure input
    #real_imgs = Variable(imgs.type(FloatTensor))
    real_imgs = Variable((y_i_2))
#         labels = Variable(labels.type(LongTensor))

    # -----------------
    #  Train Generator
    # -----------------

    optimizer_G.zero_grad()

    # Sample noise and labels as generator input
    #z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
    z = Variable(torch.cat((torch.rand(bs, rand).cuda(), sample_discrete_class(bs, 1, d_class).cuda()), 1))

    gen_labels = Variable(x_i)
    # Generate a batch of images
    gen_imgs = generator(z, gen_labels)

    # Loss measures generator's ability to fool the discriminator
    validity = discriminator(gen_imgs, gen_labels)
    g_loss = adversarial_loss(validity, valid)

    g_loss.backward()
    optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()

    labels = Variable(x_i_2)

    # Loss for real images
    validity_real = discriminator(real_imgs, labels)
    d_real_loss = adversarial_loss(validity_real, valid)

    # Loss for fake images
    validity_fake = discriminator(gen_imgs.detach(), gen_labels)
    d_fake_loss = adversarial_loss(validity_fake, fake)

    # Total discriminator loss
    d_loss = (d_real_loss + d_fake_loss) / 2

    d_loss.backward()
    optimizer_D.step()

    if j % 10000 == 0:

        #visualize_()
        sampling_process_x = []
        sampling_process_y = []
        sampling_process_true = []

        generator.train()

        for i in range(0, 1000):
            b1 = np.random.choice(x.shape[0], bs)
            x_i = torch.from_numpy(x[b1].reshape(-1, 1)).float().cuda()
            y_i = torch.from_numpy(y[b1].reshape(-1, 1)).float().cuda()

            z = Variable(torch.cat((torch.rand(bs, rand).cuda(), sample_discrete_class(bs, 1, d_class).cuda()), 1))
            gen_labels = Variable(x_i)
            gen_imgs = generator(z, gen_labels)

            sampling_process_x.append(x_i.detach().cpu().numpy())
            sampling_process_y.append(gen_imgs.detach().cpu().numpy())
            sampling_process_true.append(y_i.detach().cpu().numpy())

        sampling_process_x = np.concatenate(sampling_process_x).reshape(-1)
        sampling_process_y = np.concatenate(sampling_process_y).reshape(-1)
        sampling_process_true = np.concatenate(sampling_process_true).reshape(-1)

        plt.rcParams["figure.figsize"] = [4,4]
        
        plt.title('Iteration:{0}'.format(j))
        plt.hist2d(sampling_process_y, sampling_process_x, range=((0, 1), (0, 1)), bins=100)
        plt.xlabel('$x_{t}$', fontsize=15, labelpad = -2)
        plt.ylabel('$x_{t+1}$', fontsize=15, labelpad = -8)
        plt.tick_params(pad=3, labelsize=10)
        plt.show()