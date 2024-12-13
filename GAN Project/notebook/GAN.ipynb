# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

# -------------------------------
# Setting up hyperparameters
# -------------------------------
batchSize = 64
imageSize = 64
nz = 100  # Size of latent vector (input to the generator)
lr = 0.0002  # Learning rate
beta1 = 0.5  # Beta1 hyperparameter for Adam optimizer

# -------------------------------
# Define transformations and dataset
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((imageSize, imageSize)),  # Ensures images are resized to (64, 64)
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load CIFAR-10 dataset
dataset = torchvision.datasets.CIFAR10(
    root='./data', download=True, transform=transform
)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batchSize, shuffle=True, num_workers=2  # Added num_workers for efficiency
)

# -------------------------------
# Generator and Discriminator definitions
# -------------------------------

# Generator
class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# Discriminator
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)

# -------------------------------
# Initialize models and weights
# -------------------------------
netG = G()
netD = D()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)

# -------------------------------
# Training loop
# -------------------------------
criterion = nn.BCELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netD.to(device)
netG.to(device)
criterion.to(device)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

for epoch in range(25):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()

        # Discriminator training
        real, _ = data
        real = real.to(device)
        target_real = torch.ones(real.size(0), device=device)
        output_real = netD(real)
        errD_real = criterion(output_real, target_real)

        noise = torch.randn(real.size(0), nz, 1, 1, device=device)
        fake = netG(noise)
        target_fake = torch.zeros(real.size(0), device=device)
        output_fake = netD(fake.detach())
        errD_fake = criterion(output_fake, target_fake)

        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        # Generator training
        netG.zero_grad()
        target_gen = torch.ones(real.size(0), device=device)
        output_gen = netD(fake)
        errG = criterion(output_gen, target_gen)
        errG.backward()
        optimizerG.step()

        # Logging and saving results
        print(f'[{epoch+1}/25][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')
        if i % 100 == 0:
            vutils.save_image(real, f'./results/real_samples.png', normalize=True)
            vutils.save_image(fake.data, f'./results/fake_samples_epoch_{epoch+1:03d}.png', normalize=True)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
