import torch
import torch.nn as nn
import torch.nn.functional as F




# define a VAE
class VAE(nn.Module):
    def __init__(self, in_features=784, intermediate_dim=256, latent_dim=2):
        super(VAE, self).__init__()

        self.in_features = in_features
        self.latent_dim = latent_dim
        # encoder
        self.enc1 = nn.Linear(in_features=in_features, out_features=intermediate_dim)
        self.enc2 = nn.Linear(in_features=intermediate_dim, out_features=intermediate_dim)
        self.enc_out = nn.Linear(in_features=intermediate_dim, out_features=latent_dim + latent_dim)

        # decoder
        self.dec1 = nn.Linear(in_features=latent_dim, out_features=intermediate_dim)
        self.dec2 = nn.Linear(in_features=intermediate_dim, out_features=intermediate_dim)
        self.dec_out = nn.Linear(in_features=intermediate_dim, out_features=in_features)

        self.log_var_rec = nn.Parameter(torch.tensor(1.))

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def forward(self, x):
        # encoding
        mu_latent, log_var_latent = self.encode(x)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu_latent, log_var_latent)
        # decoding
        mu_rec = self.decode(z)
        return mu_rec, mu_latent, log_var_latent

    def encode(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = self.enc_out(x).view(-1, 2, self.latent_dim)
        # get `mu` and `log_var`
        mu_latent = x[:, 0, :]  # the first feature values as mean
        log_var_latent = x[:, 1, :]  # the other feature values as variance
        return mu_latent, log_var_latent

    def decode(self, z):
        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        mu_rec = F.sigmoid(self.dec_out(x))
        return mu_rec

    def generate_many(self, num_samples=15):
        z = torch.randn(num_samples, self.latent_dim)
        mu_rec = torch.zeros(num_samples, self.in_features)
        # decoding num_samples time
        for i in range(z.shape[0]):
            mu_rec[i] = self.decode(z[i])
        return mu_rec


