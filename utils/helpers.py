import torch
import torch.nn as nn

def weights_init(model):
    if isinstance(model, (nn.Conv2d, nn.ConvTranspose2d)):
        model.weight.data.normal_(0.0, 0.02) # xavier_normalization
    if isinstance(model, (nn.BatchNorm2d)):
        model.weight.data.normal_(0.0, 0.02)
        model.bias.data.fill_(0)

def gradient_penalty(critic, real, fake, device='cpu'):
    epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    
    mixed_scores = critic(interpolated_images)
    
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty