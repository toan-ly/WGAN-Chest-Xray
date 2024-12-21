import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, latent_dim, output_channel, hidden_dim=64):
        super(Generator, self).__init__()
        
        self.gen = nn.Sequential(
            self._block(latent_dim, hidden_dim*16, 4, 1, 0), 
            self._block(hidden_dim*16, hidden_dim*8, 4, 2, 1),
            self._block(hidden_dim*8, hidden_dim*4, 4, 2, 1),
            self._block(hidden_dim*4, hidden_dim*2, 4, 2, 1),
            self._block(hidden_dim*2, hidden_dim, 4, 2, 1),
            nn.ConvTranspose2d(hidden_dim, output_channel, 4, 2, 1),
            nn.Tanh()
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.gen(x)
    
class Discriminator(nn.Module):
    def __init__(self, input_channel, hidden_dim=64):
        super(Discriminator, self).__init__()
        
        self.disc = nn.Sequential(
            self._block(input_channel, hidden_dim, 4, 2, 1),
            self._block(hidden_dim, hidden_dim*2, 4, 2, 1),
            self._block(hidden_dim*2, hidden_dim*4, 4, 2, 1),
            self._block(hidden_dim*4, hidden_dim*8, 4, 2, 1),
            self._block(hidden_dim*8, hidden_dim*16, 4, 2, 1),
            nn.Conv2d(hidden_dim*16, 1, 4, 1, 0),
        )
            
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        return self.disc(x)
        