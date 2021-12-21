import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import model
import math
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
from torch.distributions import Independent, Normal, kl_divergence
matplotlib.style.use('ggplot')

# construct the argument parser and parser the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=10, type=int,
                    help='number of epochs to train the VAE for')
args = vars(parser.parse_args())

# learning parameters
epochs = args['epochs']
batch_size = 128
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

# train and validation data
train_data = datasets.MNIST(
    root='../input/data',
    train=True,
    download=True,
    transform=transform
)
test_data = datasets.MNIST(
    root='../input/data',
    train=False,
    download=True,
    transform=transform
)
# training and validation data loaders
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False
)

model = model.VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

def final_loss(mu_rec, logvar_rec, mu_latent, logvar_latent, orig_x, kl_weight=1):
    """
    This function will add the reconstruction loss (loss between two multivariate gaussians, in particular
    the likelihood which is the output of the decoder and the gaussian from which the input data is sampled) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    Reconstruction Loss = KL-Divergence between the 2 gaussians aforementioned = # TODO
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    logvar_rec = logvar_rec * torch.ones_like(mu_rec)
    REC_LOSS = torch.sum(logvar_rec + (orig_x - mu_rec)**2 / (2*torch.exp(logvar_rec)) + math.log(2 * np.pi))
    KLD = kl_weight * (-0.5 * torch.sum(1 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp()))
    return REC_LOSS + KLD

def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data, _ = data
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        mu_rec, mu_latent, logvar_latent = model(data)
        loss = final_loss(mu_rec, model.log_var_rec, mu_latent, logvar_latent, data)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss


# def validate(model, dataloader):
#     model.eval()
#     running_loss = 0.0
#     with torch.no_grad():
#         for i, data in tqdm(enumerate(dataloader), total=int(len(val_data) / dataloader.batch_size)):
#             data, _ = data
#             data = data.to(device)
#             data = data.view(data.size(0), -1)
#             mu_rec, mu_latent, logvar_latent = model(data)
#             loss = final_loss(mu_rec, model.log_var_rec, mu_latent, logvar_latent, data)
#             running_loss += loss.item()
#
#             # save the last batch input and output of every epoch
#             if i == int(len(val_data) / dataloader.batch_size) - 1:
#                 num_rows = 8
#                 both = torch.cat((data.view(batch_size, 1, 28, 28)[:8],
#                                   mu_rec.view(batch_size, 1, 28, 28)[:8]))
#                 save_image(both.cpu(), f"outputs/output{epoch}.png", nrow=num_rows)
#     val_loss = running_loss / len(dataloader.dataset)
#     return val_loss

def test(model, dataloader, epoch=None, save=False):
    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(test_data) / dataloader.batch_size)):
            data, label = data
            data = data.view(data.size(0), -1)
            z = model.get_latent_representation(data)
            if z.shape[1] > 2:
                print("Attention, latent space dimension is higher than 2..")
            plt.scatter(z[:, 0], z[:, 1], c=label, cmap='tab10')
        plt.colorbar()
        plt.grid(False)
        if save:
            plt.savefig(f"outputs/latent_space/latent_space{epoch}.png")
            plt.clf()
        else:
            plt.show()

def plot_reconstructed(model, epoch=None, r0=(-5, 10), r1=(-10, 5), n=12, save=False):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = model.generate(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    if save:
        save_image(torch.Tensor(img), f"outputs/latent_space_digits/latent_space{epoch}.png")
    else:
        plt.imshow(img, extent=[*r0, *r1])
        plt.grid(False)
        plt.show()

def plot_loss(train_loss, epochs):
    x = train_loss
    plt.plot(x, label='train_loss', c='blue')
    plt.title("ELBO LOSS PLOT")
    plt.legend()
    plt.show()

train_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs} - {model.log_var_rec}")
    train_epoch_loss = fit(model, train_loader)
    train_loss.append(train_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    plot_reconstructed(model, epoch=epoch, save=True)
    test(model, test_loader, epoch=epoch, save=True)


num_samples = 15
generated_digits = model.generate_many(num_samples=num_samples)
save_image(generated_digits.view(num_samples, 1, 28, 28), f"generated.png", nrow=num_samples)



plot_loss(train_loss, epochs)