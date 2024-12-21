import glob
import os
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_samples_per_class=1000):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        for label, subdir in enumerate(['PNEUMONIA', 'NORMAL']):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.exists(subdir_path):
                all_images = [os.path.join(subdir_path, file) for file in os.listdir(subdir_path) if file.endswith(('jpeg', 'png', 'jpg', 'bmp', 'tiff'))]
                
                if len(all_images) > num_samples_per_class:
                    all_images = random.sample(all_images, num_samples_per_class)
                self.image_files.extend(all_images)
                self.labels.extend([label] * len(all_images))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
def load_data(data_dir, batch_size, im_channels, im_size):
    # Define transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=im_channels),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(im_channels)], [0.5 for _ in range(im_channels)])
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return data_loader