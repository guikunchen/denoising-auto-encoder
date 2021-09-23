import torch
import torchvision.transforms as transforms
import torchvision
from model.conv_ae import ConvAE
from utils.ssim import SSIM

checkpoint_path = "checkpoints/ae799.pt"
# fname = "noise_img/xihu0.4.png"
# denoising_fname = "denoise_img/xihu0.4.png"
# fname = "/home/gkc/projects/CNNDenoisingAutoEncoder/noise_img/0000000232720.4.jpg"
fname = "/home/gkc/projects/CNNDenoisingAutoEncoder/normal_img/000000023272.jpg"
denoising_fname = "denoise_img/0000000232720.4.jpg"

if __name__ == '__main__':
    # model = ConvAE()
    # model.load_state_dict(torch.load(checkpoint_path))
    model = torch.load(checkpoint_path)
    model.eval()

    compose = [
        transforms.ToPILImage(),
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)

    # 1. Load the image
    img = torchvision.io.read_image(fname)
    height, width = img.shape[1], img.shape[2]
    # 2. normalize the images using torchvision.
    img = transform(img).unsqueeze(0).cuda()

    noise_imgs = img.clone().detach().cpu().numpy()
    noise_imgs = torch.from_numpy(binary_noise_mask_image(noise_imgs, 0.8)).cuda().float()

    denoising_img = model(noise_imgs)
    ssim = SSIM()
    print(ssim(denoising_img, img))
    denoising_img = (denoising_img + 1) / 2.  # restore to [0, 1]
    # resize to original
    detransform = transforms.Compose([transforms.Resize((height, width))])
    denoising_img = detransform(denoising_img)
    
    torchvision.utils.save_image(denoising_img, denoising_fname)
