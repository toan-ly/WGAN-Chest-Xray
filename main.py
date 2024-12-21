import os
import torch
import torch.optim as optim

from data.loader import load_data
from models.models import Generator, Discriminator
from utils.helpers import weights_init, gradient_penalty
from training.train import train
from config import *


def main():
    data_dir = './input/chest-xray-pneumonia/chest_xray'
    dataloader = load_data(data_dir)
    
    gen = Generator(latent_dim, im_channels).to(device)
    gen_optim = optim.Adam(gen.parameters(), lr=learning_rate, betas=betas)
    gen = gen.apply(weights_init)
    
    critic = Discriminator(im_channels).to(device)
    critic_optim = optim.Adam(critic.parameters(), lr=learning_rate, betas=betas)
    critic = critic.apply(weights_init)
    
    train(gen, critic, gen_optim, critic_optim, dataloader, config=locals(), device=device)
    
    
    
    