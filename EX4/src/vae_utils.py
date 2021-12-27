import torch
import torch.nn as nn
from torchvision.utils import save_image

import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def elbo_loss(CV_loss, mu_latent, logvar_latent):
    """
    This function will add the reconstruction loss (cross entropy as difference between two probability ditributions)
    and the KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param CV_loss: loss value given by criterion
    :param mu_latent: the mean from the latent vector
    :param logvar_latent: log variance from the latent vector
    """
    REC_LOSS = CV_loss
    KLD = (-0.5 * torch.sum(1 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp()))
    return KLD, REC_LOSS, REC_LOSS + KLD


def elbo_loss_alternative(mu_rec, logvar_rec, mu_latent, logvar_latent, orig_x, kl_weight=1):
    """
    An alternative which tries to achieve the same underlying goal, in a more mathematical and hands-one manner
    This function will add the reconstruction loss (log likelihood of output distribution for the input data) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    Reconstruction Loss =  - loglikelihood over normal distribution given a sample (orig_x)
        = sum(0.5 * log(variance) + (sample - mu)^2 / (2 * variance^2) + log(2pi))
    :param mu_rec: decoder output, mean of reconstruction
    :param logvar_rec: globally trainable parameter, log_variance of reconstruction
    :param mu_latent: the mean from the latent vector
    :param logvar_latent: log variance from the latent vector
    :param orig_x: input data
    :param kl_weight: possible variant, weight for the KLD so to give an arbitrary weight to that part of loss
    """
    logvar_rec = logvar_rec * torch.ones_like(mu_rec)  # replicate scalar logvar_rec to create diagonal shaped as mu_rec
    REC_LOSS = torch.sum(0.5 * (logvar_rec + (orig_x - mu_rec) ** 2 / (torch.exp(logvar_rec)) + math.log(2 * np.pi)))
    KLD = kl_weight * (-0.5 * torch.sum(1 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp()))
    return KLD, REC_LOSS, REC_LOSS + KLD


def fit(model, dataloader, optimizer, train_data, labelled=True, use_BCE=True):
    """
    fit method using Cross Entropy loss, executes one epoch
    :param model: VAE model to train
    :param dataloader: input dataloader to fatch batches
    :param optimizer: which optimizer to utilize
    :param train_data: useful for plotting completion bar
    :param labelled: to know if the data is composed of (data, target) or only data
    :param use_BCE: if True then use BCELoss, otherwise use MSELoss
    :return: train loss
    """
    model.train()  # set in train mode
    running_loss, running_kld_loss, running_rec_loss = 0.0, 0.0, 0.0  # set up losses to accumulate over
    criterion = nn.BCELoss(reduction='sum') if use_BCE else nn.MSELoss(reduction='sum')  # set up criterion for loss
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)):
        data = data[0] if labelled else data  # get the train batch
        data = data.view(data.size(0), -1)  # unroll
        data = data.float()
        optimizer.zero_grad()  # set gradient to zero
        mu_rec, mu_latent, logvar_latent = model(data)  # feedforward
        CV_loss = criterion(mu_rec, data)  # get rec loss value
        loss = elbo_loss(CV_loss, mu_latent, logvar_latent)  # get loss value
        # update losses
        running_kld_loss += loss[0].item()
        running_rec_loss += loss[1].item()
        running_loss += loss[2].item()
        loss[2].backward()  # set up gradient with total loss
        optimizer.step()  # backprop
    # set up return variable for all three losses
    train_loss = [running_kld_loss / len(dataloader.dataset),
                  running_rec_loss / len(dataloader.dataset),
                  running_loss / len(dataloader.dataset)]
    return train_loss


def test(model, dataloader, test_data, labelled=None, epoch=None, save=False, plot=True):
    """
    test method using Cross Entropy loss, over all dataset, retrieve relevant info if requested
    :param model: VAE model to test
    :param dataloader: input dataloader to fatch batches
    :param test_data: useful for plotting completion bar
    :param labelled: to know if the data is composed of (data, target) or only data
    :param epoch: for save file name
    :param save: to know if needed to save the plot or not
    :param plot: to know if needed to plot or not
    :return: test loss
    """
    model.eval()  # set in eval mode
    running_loss, running_kld_loss, running_rec_loss = 0.0, 0.0, 0.0  # set up losses to accumulate over
    criterion = nn.BCELoss(reduction='sum')  # set up criterion for loss
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(test_data) / dataloader.batch_size)):
            # fetch batch
            if labelled is not None:
                data, label = data
            data = data.view(data.size(0), -1)
            data = data.float()
            mu_rec, mu_latent, logvar_latent = model(data)  # feedforward
            CV_loss = criterion(mu_rec, data)  # get rec loss
            loss = elbo_loss(CV_loss, mu_latent, logvar_latent)  # get total loss
            # update losses
            running_kld_loss += loss[0].item()
            running_rec_loss += loss[1].item()
            running_loss += loss[2].item()
            # reconstruct digits and construct latent space (only if the task is the correct one)
            if labelled is not None:
                if i == int(len(test_data) / dataloader.batch_size) - 1:
                    save_reconstructed(data, mu_rec, epoch, batch_size=dataloader.batch_size)
                if mu_latent.shape[1] > 2:
                    continue
                plt.scatter(mu_latent[:, 0], mu_latent[:, 1], c=label, cmap='tab10', s=5)
        # if the task allows for it, if the user wants to plot then plot latent space
        if labelled is not None:
            if plot:
                if mu_latent.shape[1] <= 2:
                    plt.colorbar()
                    plt.grid(False)
                    # save instead of plotting if requested
                    if save:
                        plt.savefig(f"../outputs/latent_space/latent_space{epoch}.png")
                        plt.clf()
                    else:
                        plt.show()
    # set up return variable for all three losses
    test_loss = [running_kld_loss / len(dataloader.dataset),
                 running_rec_loss / len(dataloader.dataset),
                 running_loss / len(dataloader.dataset)]
    return test_loss


def fit_alternative(model, dataloader, optimizer, train_data, labelled=True):
    """
    fit method using alternative loss, executes one epoch
    :param model: VAE model to train
    :param dataloader: input dataloader to fatch batches
    :param optimizer: which optimizer to utilize
    :param train_data: useful for plotting completion bar
    :param labelled: to know if the data is composed of (data, target) or only data
    :return: train loss
    """
    model.train()  # set in train mode
    running_loss, running_kld_loss, running_rec_loss = 0.0, 0.0, 0.0  # set up losses to accumulate over
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)):
        data = data[0] if labelled else data  # get the train batch
        data = data.view(data.size(0), -1)  # unroll
        optimizer.zero_grad()  # set gradient to zero
        mu_rec, mu_latent, logvar_latent = model(data)  # feedforward
        loss = elbo_loss_alternative(mu_rec, model.log_var_rec, mu_latent, logvar_latent, data)  # get loss value
        # update losses
        running_kld_loss += loss[0].item()
        running_rec_loss += loss[1].item()
        running_loss += loss[2].item()
        loss[2].backward()  # set up gradient with total loss
        optimizer.step()  # backprop
    # set up return variable for all three losses
    train_loss = [running_kld_loss / len(dataloader.dataset),
                  running_rec_loss / len(dataloader.dataset),
                  running_loss / len(dataloader.dataset)]
    return train_loss


def test_alternative(model, dataloader, test_data, labelled=None, epoch=None, save=False, plot=True):
    """
    test method using Cross Entropy loss, over all dataset, retrieve relevant info if requested
    :param model: VAE model to test
    :param dataloader: input dataloader to fatch batches
    :param test_data: useful for plotting completion bar
    :param labelled: to know if the data is composed of (data, target) or only data
    :param epoch: for save file name
    :param save: to know if needed to save the plot or not
    :param plot: to know if needed to plot or not
    :return: test loss
    """
    model.eval()  # set in eval mode
    running_loss, running_kld_loss, running_rec_loss = 0.0, 0.0, 0.0  # set up losses to accumulate over
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(test_data) / dataloader.batch_size)):
            # fetch batch
            if labelled is not None:
                data, label = data
            data = data.view(data.size(0), -1)
            mu_rec, mu_latent, logvar_latent = model(data)  # feedforward
            loss = elbo_loss_alternative(mu_rec, model.log_var_rec, mu_latent, logvar_latent, data)  # get total loss
            # update losses
            running_kld_loss += loss[0].item()
            running_rec_loss += loss[1].item()
            running_loss += loss[2].item()
            # reconstruct digits and construct latent space (only if the task is the correct one)
            if labelled is not None:
                if i == int(len(test_data) / dataloader.batch_size) - 1:
                    save_reconstructed(data, mu_rec, epoch, batch_size=dataloader.batch_size)
                if mu_latent.shape[1] > 2:
                    continue
                plt.scatter(mu_latent[:, 0], mu_latent[:, 1], c=label, cmap='tab10', s=5)
        # if the task allows for it, if the user wants to plot then plot latent space
        if labelled is not None:
            if plot:
                if mu_latent.shape[1] <= 2:
                    plt.colorbar()
                    plt.grid(False)
                    # save instead of plotting if requested
                    if save:
                        plt.savefig(f"../outputs/latent_space/latent_space{epoch}.png")
                        plt.clf()
                    else:
                        plt.show()
    # set up return variable for all three losses
    test_loss = [running_kld_loss / len(dataloader.dataset),
                 running_rec_loss / len(dataloader.dataset),
                 running_loss / len(dataloader.dataset)]
    return test_loss


def plot_reconstructed_digits(model, epoch=None, r0=(-8, 8), r1=(-8, 8), n=30, save=False):
    """
    plot figure showing digits distribution over the latent space
    :param model: the VAE model to use
    :param epoch: epoch num, needed for save file name
    :param r0: defines the ensemble of values to try on first axes
    :param r1: defines the ensemble of values to try on second axes
    :param n: how many samples on each axes
    :param save: if to save the result to file
    :return:
    """
    # if latent space is not 2D, get out!
    if model.latent_dim != 2:
        return
    w = 28  # we know the dimensionality row, column
    img = np.zeros((n * w, n * w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]])  # defined the "sampled" value
            x_hat = model.decode(z)  # decode the sample
            x_hat = x_hat.reshape(28, 28).detach().numpy()
            img[(n - 1 - i) * w:(n - 1 - i + 1) * w, j * w:(j + 1) * w] = x_hat  # put it in the image
    # save the image if requested, otherwise plot it
    if save:
        save_image(torch.Tensor(img), f"../outputs/latent_space_digits/latent_space{epoch}.png")
    else:
        plt.imshow(img, extent=[*r0, *r1])
        plt.grid(False)
        plt.show()


def plot_loss(train_loss=None, test_loss=None, epochs=None, figsize=(10, 10)):
    """
    Plot the 3 losses (KLD, REC_LOSS, REC_LOSS + KLD) for possibly train and test data
    :param train_loss: array where elements are [KLD, REC_LOSS, REC_LOSS + KLD]
    :param test_loss: array where elements are [KLD, REC_LOSS, REC_LOSS + KLD]
    :param epochs: number of epochs for x axis
    :param figsize: plotted window width, height
    :return:
    """
    fig, axs = plt.subplots(2, 1, figsize=figsize)
    # plot train loss if given
    if train_loss is not None:
        rec_loss = [x[1] for x in train_loss]
        total_loss = [x[2] for x in train_loss]
        kld_loss = [x[0] for x in train_loss]
        axs[0].plot(rec_loss, label='rec_train_loss')
        axs[0].plot(total_loss, label='total_train_loss')
        axs[1].plot(kld_loss, label='kld_train_loss')
    # plot test loss if given
    if test_loss is not None:
        rec_loss = [x[1] for x in test_loss]
        total_loss = [x[2] for x in test_loss]
        kld_loss = [x[0] for x in test_loss]
        axs[0].plot(rec_loss, label='rec_test_loss')
        axs[0].plot(total_loss, label='total_test_loss')
        axs[1].plot(kld_loss, label='kld_test_loss')
    plt.title("ELBO LOSS PLOT")
    axs[0].legend()
    axs[1].legend()
    axs[0].set_title("total and reconstructed loss")
    axs[1].set_title("kld loss")
    plt.show()


def save_reconstructed(data, mu_rec, epoch, batch_size):
    """
    takes the first 15 outputs from the batch, couples them with the respective input and creates an image of
    original vs reconstructed data, saving it
    :param data: original data batch
    :param mu_rec: output data batch
    :param epoch: needed for save file name
    :param batch_size: batch size of given data and mu_rec
    """
    # save the last batch input and output of every epoch
    num_rows = 15
    img = torch.cat((data.view(batch_size, 1, 28, 28)[:15], mu_rec.view(batch_size, 1, 28, 28)[:15]))
    save_image(img, f"../outputs/reconstructed_vs_original/output{epoch}.png", nrow=num_rows)


def get_MI_reconstruction(model, dataloader, dataset):
    """
    to be called once the model is trained, gets the dataset reconstruction in a single numpy array (batched)
    :param model: model to use as feedforward
    :param dataloader: useful to iter through data
    :param dataset: data to use
    """
    model.eval()
    reconstruction = []
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
        data = data.view(data.size(0), -1)
        data = data.float()
        mu_rec, mu_latent, logvar_latent = model(data)
        reconstruction.append(mu_rec.detach())
    return reconstruction


def check_number_in_box(data, rect_diag_0, rect_diag_1, counter=0):
    """
    function to retrieve how many points are in a given rectangle
    :param data: data to check
    :param rect_diag_0: coordinates of one side of the diagonal
    :param rect_diag_1: coordinates of the other side of the diagonal
    :param counter: can possibly start not from 0
    :return: counter of how many people are in rectangle + counter
    """
    for elem in data:
        if rect_diag_0[0] <= elem[0] <= rect_diag_1[0] and rect_diag_1[1] <= elem[1] <= rect_diag_0[1]:
            counter += 1
    return counter
