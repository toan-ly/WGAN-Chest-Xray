import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch

def plot_images(fake, epoch, save_path):
    im_grid_fake = make_grid(fake[:16], nrow=4, normalize=True)
    plt.imshow(im_grid_fake.permute(1, 2, 0).squeeze().cpu())
    plt.axis("off")
    plt.title(f"Generated Images at Epoch {epoch + 1}")
    plt.savefig(save_path)
    plt.show()

def plot_multiple_img(img_matrix_list, titles_list, ncols=4, main_title="Different Types of Augmentations"):
    n_images = len(img_matrix_list)
    nrows = (n_images + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < n_images:
            img = img_matrix_list[i]
            if isinstance(img, torch.Tensor):  # Handle torch tensors.
                img = img.permute(1, 2, 0).squeeze().cpu().numpy()  # Convert to numpy.
            axes[i].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[i].set_title(titles_list[i], fontsize=10)
        axes[i].axis('off') 

    for i in range(n_images, len(axes)):
        axes[i].axis('off')  

    fig.suptitle(main_title, fontsize=16, weight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.88) 
    plt.show()
