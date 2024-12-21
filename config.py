import torch

latent_dim = 100
im_channels = 1
im_size = 128

epochs = 150
batch_size = 32
learning_rate = 2e-4
lambda_gp = 10
n_critic = 5
betas = (0.3, 0.99)

device = "cuda" if torch.cuda.is_available() else "cpu"
display_epoch = 10

result_dir = "./results"