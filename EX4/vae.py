import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Class implementing the architecture of a Variational AutoEncoder, as well as a method to generate multiple samples
    """
    def __init__(self, in_features=784, intermediate_dim=256, latent_dim=2, use_BCE_loss = True):
        """
        initialization of the model
        :param in_features: input dimension of the data, fed to input layer (ignoring batch_size)
        :param intermediate_dim: dimension of the hidden layers, define the feature space dimensionality intramodel
        :param latent_dim: dimension of the latent space, dimensionality for both the statistic measures of the
        latent space distribution
        :param use_BCE_loss: decides if last layer of decoding has to go through sigmoid or not
        """
        super(VAE, self).__init__()

        self.in_features = in_features
        self.latent_dim = latent_dim
        self.use_BCE_loss = use_BCE_loss

        # set up the encoder - fetch the input, elaborate, output the mean and logvar
        self.enc1 = nn.Linear(in_features=in_features, out_features=intermediate_dim)
        self.enc2 = nn.Linear(in_features=intermediate_dim, out_features=intermediate_dim)
        self.enc_out = nn.Linear(in_features=intermediate_dim, out_features=latent_dim + latent_dim)

        # set up the decoder - fetch a sample from the latent space (which is conditioned on the input distribution),
        # elaborate, output the mean of the output probability distribution
        self.dec1 = nn.Linear(in_features=latent_dim, out_features=intermediate_dim)
        self.dec2 = nn.Linear(in_features=intermediate_dim, out_features=intermediate_dim)
        self.dec_out = nn.Linear(in_features=intermediate_dim, out_features=in_features)

        # set up the reconstruction logvar for the output distribution as a globally trainable parameter
        self.log_var_rec = nn.Parameter(torch.tensor(1.))  # by default requires_grad = True

    def reparameterize(self, mu, log_var):
        """
        applies the reparameterization trick to allow backprop to flow to encoder part of the model
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def forward(self, x):
        """
            feedforward
            :param x: input data
            :return: mean of reconstructed data,
            mean of latent space,
            log_var of latent space
        """
        mu_latent, log_var_latent = self.encode(x)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu_latent, log_var_latent)
        # decoding
        mu_rec = self.decode(z)
        return mu_rec, mu_latent, log_var_latent

    def encode(self, x):
        """
        encoding feedforward
        :param x: input data
        :return: mean of latent space, log_var of latent space
        """
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = self.enc_out(x).view(-1, 2, self.latent_dim)
        # get `mu` and `log_var`
        mu_latent = x[:, 0, :]  # the first feature values as mean
        log_var_latent = x[:, 1, :]  # the other feature values as variance
        return mu_latent, log_var_latent

    def decode(self, z):
        """
        decoding feedforward
        :param z: sampled value from latent space distribution
        :return: mean of reconstructed data
        """
        # decoding feedforward
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        if self.use_BCE_loss:
            mu_rec = F.sigmoid(self.dec_out(x))  # apply sigmoid if using BCELoss otherwise not needed
        else:
            mu_rec = self.dec_out(x)
        return mu_rec

    def generate_many(self, num_samples=15):
        """
        sample from created latent space a certain number of time, then decode
        :param num_samples: how many samples to fetch
        :return: batch_size of reconstructed means
        """
        z = torch.randn(num_samples, self.latent_dim)
        mu_rec = torch.zeros(num_samples, self.in_features)
        # decoding num_samples time
        for i in range(z.shape[0]):
            mu_rec[i] = self.decode(z[i])
        return mu_rec


