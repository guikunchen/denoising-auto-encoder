import os
import glob

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from utils.noise import binary_noise_mask_image


class CommonImageDataset(Dataset):
    def __init__(self, fnames, transform, noise_ratio):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)
        self.noise_ratio = noise_ratio

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        # 1. Load the image
        raw_img = torchvision.io.read_image(fname)
        noise_img = raw_img.clone().detach().cpu().numpy()
        noise_img = torch.Tensor(binary_noise_mask_image(noise_img, noise_ratio=self.noise_ratio, batch=False))
        # 2. normalize the images using torchvision.
        raw_img = self.transform(raw_img)
        noise_img = self.transform(noise_img)
        return raw_img, noise_img

    def __len__(self):
        return self.num_samples


def _get_dataset(root, noise_ratio, img_size):
    fnames = glob.glob(os.path.join(root, '*'))
    # Linearly map [0, 1] to [-1, 1]
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        transforms.ToTensor(),  # [0, 1] here
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = CommonImageDataset(fnames, transform, noise_ratio)
    return dataset


def get_dataloader(path, batch_size, noise_ratio, img_size=1024):
    dataset = _get_dataset(path, noise_ratio, img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


if __name__ == '__main__':
    dataset = _get_dataset(os.path.join(workspace_dir, 'faces'))

    images = [dataset[i] for i in range(16)]
    grid_img = torchvision.utils.make_grid(images, nrow=4)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()

    images = [(dataset[i]+1)/2 for i in range(16)]
    grid_img = torchvision.utils.make_grid(images, nrow=4)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
