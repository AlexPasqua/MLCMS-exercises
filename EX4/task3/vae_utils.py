import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import vae
import math
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np


def elbo_loss(mu_rec, logvar_rec, mu_latent, logvar_latent, orig_x, kl_weight=1):
    """
    This function will add the reconstruction loss (loss between two multivariate gaussians, in particular
    the likelihood which is the output of the decoder and the gaussian from which the input data is sampled) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    Reconstruction Loss = KL-Divergence between the 2 gaussians aforementioned = # TODO
    :param mu_rec: decoder output, mean of reconstruction
    :param logvar_rec: globally trainable parameter, log_variance of reconstruction
    :param mu_latent: the mean from the latent vector
    :param logvar_latent: log variance from the latent vector
    :param orig_x: the mean of the input data
    :param kl_weight: possible variant, weight for the KLD so to give an arbitrary weight to that part of loss
    """
    logvar_rec = logvar_rec * torch.ones_like(mu_rec)  # replicate logvar_rec to create diagonal shaped as mu_rec
    REC_LOSS = torch.sum(logvar_rec + (orig_x - mu_rec)**2 / (2*torch.exp(logvar_rec)) + math.log(2 * np.pi))
    KLD = kl_weight * (-0.5 * torch.sum(1 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp()))
    return REC_LOSS + KLD


def fit(model, dataloader, optimizer, train_data, epoch=None, labelled=True):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data = data[0] if labelled else data
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        mu_rec, mu_latent, logvar_latent = model(data)
        loss = elbo_loss(mu_rec, model.log_var_rec, mu_latent, logvar_latent, data)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss


def test(model, dataloader, test_data, epoch=None, save=False):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(test_data) / dataloader.batch_size)):
            data, label = data
            data = data.view(data.size(0), -1)
            mu_rec, mu_latent, logvar_latent = model(data)
            loss = elbo_loss(mu_rec, model.log_var_rec, mu_latent, logvar_latent, data)
            running_loss += loss.item()
            if i == int(len(test_data) / dataloader.batch_size) - 1:
                save_reconstructed(data, mu_rec, epoch, batch_size=dataloader.batch_size)
            if mu_latent.shape[1] > 2:
                continue
            plt.scatter(mu_latent[:, 0], mu_latent[:, 1], c=label, cmap='tab10', s=5)
        if mu_latent.shape[1] <= 2:
            plt.colorbar()
            plt.grid(False)
            if save:
                plt.savefig(f"outputs/latent_space/latent_space{epoch}.png")
                plt.clf()
            else:
                plt.show()
    test_loss = running_loss / len(dataloader.dataset)
    return test_loss


def plot_reconstructed_digits(model, epoch=None, r0=(-8, 8), r1=(-8, 8), n=30, save=False):
    if model.latent_dim != 2:
        return
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]])
            x_hat = model.decode(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    if save:
        save_image(torch.Tensor(img), f"outputs/latent_space_digits/latent_space{epoch}.png")
    else:
        plt.imshow(img, extent=[*r0, *r1])
        plt.grid(False)
        plt.show()


def plot_loss(train_loss, test_loss, epochs):
    x = train_loss
    plt.plot(x, label='train_loss', c='blue')
    x = test_loss
    plt.plot(x, label='test_loss', c='red', linestyle='dashed')
    plt.title("ELBO LOSS PLOT")
    plt.legend()
    plt.show()


def save_reconstructed(data, mu_rec, epoch, batch_size):
    # save the last batch input and output of every epoch
    num_rows = 15
    img = torch.cat((data.view(batch_size, 1, 28, 28)[:15], mu_rec.view(batch_size, 1, 28, 28)[:15]))
    save_image(img, f"outputs/reconstructed_vs_original/output{epoch}.png", nrow=num_rows)