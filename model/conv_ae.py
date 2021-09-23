import torch
import torch.nn as nn

class ConvAE(nn.Module):

    def __init__(self):
        super(ConvAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),         
            nn.LeakyReLU(0.1),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),        
            nn.LeakyReLU(0.1),
			      nn.Conv2d(24, 48, 4, stride=2, padding=1),         
            nn.LeakyReLU(0.1),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
			      nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1), 
            nn.LeakyReLU(0.1),
			      nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1), 
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
