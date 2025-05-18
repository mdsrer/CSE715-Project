import torch.nn as nn
import torch
import numpy as np
from torchviz import make_dot
import graphviz
from graphviz import Digraph

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128, in_channels=3, input_size=(288, 432)):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Flatten()
        )
        
        # Calculate the size after flattening
        self.flat_size = self._get_flat_size(in_channels)
        
        # VAE latent space
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flat_size)
        
        # Calculate spatial dimensions after flattening
        h, w = self.input_size
        h_out = h // 16  # After 4 stride-2 convolutions
        w_out = w // 16
        self.decoder_spatial_h = h_out
        self.decoder_spatial_w = w_out
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, self.decoder_spatial_h, self.decoder_spatial_w)),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def _get_flat_size(self, in_channels):
        # Create a dummy input to calculate the size after flattening
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, self.input_size[0], self.input_size[1])
            x = self.encoder(dummy_input)
            return x.shape[1]  # Return the flattened dimension
    
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        x = self.decoder_input(z)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, z, mu, logvar