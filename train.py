import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

from utils.torch_utils import select_device
from utils.random_seeds import same_seeds
from data.dataloader import get_dataloader
from model.conv_ae import ConvAE
# from model.vae import VAE, loss_vae

lr = 1e-3
image_path = "/home/gkc/dataset/image_restoration"
img_size = 1024


def train(opt, device, log_dir, ckpt_dir):
    # Model
    # model = ConvAE().to(device)
    # model.load_state_dict(torch.load("oldcheckpoints/cnn/0.pt").state_dict())
    model = torch.load("oldcheckpoints/cnn/adam800epch.pt").to(device)
    model.train()

    # loss and Optimizer
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # train loop
    # store some pictures regularly to monitor the current performance of the Generator,
    # and regularly record checkpoints.
    noise_ratio = 0.
    for e, epoch in enumerate(range(opt.n_epoch)):
        dataloader = get_dataloader(image_path, opt.batch_size, noise_ratio, img_size)
        tot_loss = list()
        for _, (raw_imgs, noise_imgs) in enumerate(dataloader):
            raw_imgs = raw_imgs.to(device)
            noise_imgs = noise_imgs.to(device)

            # denoising_imgs, mu, logvar = model(noise_imgs)
            denoising_imgs = model(noise_imgs)

            # loss = loss_vae(denoising_imgs, raw_imgs, mu, logvar, criterion)
            loss = criterion(denoising_imgs, raw_imgs)

            tot_loss.append(loss.item())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        mean_loss = np.mean(tot_loss)
        print("Epoch: " + str(e+1) + " Loss: " + str(round(mean_loss, 4)) + " Ratio: " + str(round(noise_ratio, 2)))

        if (e + 1) % 50 == 0 or e == 0:
            # Save the checkpoints.
            torch.save(model, os.path.join(ckpt_dir, "ae" + str(e) + ".pt"))

            # Save results
            model.eval()
            noise_imgs = (noise_imgs + 1) / 2.  # restore to [0, 1]
            filename = os.path.join(log_dir, f'Epoch_{epoch + 1:03d}_noise{round(noise_ratio, 2)}.jpg')
            torchvision.utils.save_image(noise_imgs, filename, nrow=10)

            denoising_imgs = (denoising_imgs + 1) / 2.  # restore to [0, 1]
            filename = os.path.join(log_dir, f'Epoch_{epoch + 1:03d}_loss{round(mean_loss, 4)}.jpg')
            torchvision.utils.save_image(denoising_imgs, filename, nrow=10)
            print(f' | Save some samples to {filename}.')
            model.train()

        noise_ratio = get_noise_ratio(e)


def get_noise_ratio(e):
    noise_ratio = 0.
    if e < 40:
        noise_ratio = 0
    elif e < 80:
        noise_ratio = 0.4
    elif e < 130:
        noise_ratio = 0.6
    elif e < 190:
        noise_ratio = 0.8
    elif e < 250:
        noise_ratio = 0.6
    elif e < 300:
        noise_ratio = 0.9
    elif e < 360:
        noise_ratio = 0.8
    else:
        noise_ratio = 0.7
    
    return noise_ratio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace_dir', nargs='+', type=str, default='/home/gkc/projects/CNNDenoisingAutoEncoder',
                        help='workspace path(s)')
    parser.add_argument('--n-epoch', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    # set random seeds
    same_seeds(2021)

    # set dirs of logs and checkpoints
    log_dir = os.path.join(opt.workspace_dir, 'logs')
    ckpt_dir = os.path.join(opt.workspace_dir, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # select_device
    device = select_device(opt.device, batch_size=opt.batch_size)

    # train
    train(opt, device, log_dir, ckpt_dir)
