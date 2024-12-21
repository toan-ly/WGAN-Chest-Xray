from tqdm.auto import tqdm
import torch
from utils.helpers import gradient_penalty
from utils.visualize import plot_images
import os

def train(gen, critic, gen_optim, critic_optim, dataloader, config, device):
    gen_losses, critic_losses = [], []
    fixed_noise = torch.randn((config.batch_size, config.latent_dim, 1, 1)).to(device)

    for epoch in range(config.epochs):
        batch_idx = 0
        print(f'[INFO] training epoch {epoch+1}...')

        for real, _ in tqdm(dataloader):
            real = real.to(device)
            mean_critic_loss = 0

            # Train Critic
            for _ in range(config.n_critic):
                noise = torch.randn((config.batch_size, config.latent_dim, 1, 1)).to(device)
                fake = gen(noise)

                critic_real = critic(real)
                critic_fake = critic(fake)
                gp = gradient_penalty(critic, real, fake, device)
                critic_loss = -(torch.mean(critic_real) - torch.mean(critic_fake)) + config.lambda_gp * gp

                critic.zero_grad()
                critic_loss.backward(retain_graph=True)
                critic_optim.step()

                mean_critic_loss += critic_loss.item() / config.n_critic

            critic_losses.append(mean_critic_loss)

            # Train Generator
            noise = torch.randn((config.batch_size, config.latent_dim, 1, 1)).to(device)
            fake = gen(noise)
            output = critic(fake)
            gen_loss = -torch.mean(output)

            gen.zero_grad()
            gen_loss.backward()
            gen_optim.step()

            gen_losses.append(gen_loss.item())
            
            batch_idx += 1
            
        if (epoch + 1) % config.display_epoch == 0:
            print(f"[{epoch+1}/{config.epochs}][{batch_idx}/{len(dataloader)}] \
                    Loss D: {critic_loss:.4f}, loss G: {gen_loss:.4f}")
            with torch.no_grad():
                fake = gen(fixed_noise)
                plot_images(fake, epoch, os.path.join(config.result_dir, 'image_at_epoch_{:04d}.png'.format(epoch+1)))
        