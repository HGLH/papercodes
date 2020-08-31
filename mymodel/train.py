import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from model import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)
input_shape = (opt.channels, opt.img_height, opt.img_width)
# Initialize generator and discriminator
generator =GeneratorUNet()
discriminator_s= Discriminator_S(input_shape)
discriminator_c=Discriminator_C()
if cuda:
    generator = generator.cuda()
    discriminator_s = discriminator_s.cuda()
    discriminator_c= discriminator_c.cuda()
    criterion_GAN.cuda()

if opt.epoch != 0:
    # Load pretrained models
    print("加载pre_trained模型")
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator_s.load_state_dict(torch.load("saved_models/%s/discriminators_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator_c.load_state_dict(torch.load("saved_models/%s/discriminatorc_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator_s.apply(weights_init_normal)
    discriminator_c.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_DS = torch.optim.Adam(discriminator_s.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_DC = torch.optim.Adam(discriminator_c.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# Configure dataloaders
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset("./datasets/%s" % opt.dataset_name, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    ImageDataset("./datasets/%s" % opt.dataset_name, transforms_=transforms_, mode="val"),
    batch_size=10,
    shuffle=True,
    num_workers=1,
    drop_last=True,
)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data), -2)
    save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)


# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Model inputs
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()
        # GAN loss
        fake_B = generator(real_A)
        pred_sfake = discriminator_s(fake_B)
        pred_cfake=discriminator_s(real_B)

        loss_GAN1 = criterion_GAN(pred_sfake, valid)
        loss_GAN2 = criterion_GAN(pred_cfake,valid)

        # Total loss
        loss_G = loss_GAN1 + loss_GAN2
        #loss_GAN1 + lambda*pixel * loss_GAN2可以用lambda优化

        loss_G.backward(retain_graph=True)

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_DS.zero_grad()
        # Real loss
        pred_sreal = discriminator_s(real_B)
        loss_sreal = criterion_GAN(pred_sreal, valid)

        # Fake loss
        pred_sfake = discriminator_s(fake_B.detach())
        loss_sfake = criterion_GAN(pred_sfake, fake)

        # Total loss
        loss_DS = 0.5 * (loss_sreal + loss_sfake)

        loss_DS.backward()
        optimizer_DS.step()

        optimizer_DC.zero_grad()

        # Real loss
        pred_creal = discriminator_c(real_A, fake_B)
        loss_creal = criterion_GAN(pred_creal, valid)

        # Fake loss
        pred_cfake = discriminator_c(real_A,real_B)
        loss_cfake = criterion_GAN(pred_cfake, fake)

        # Total loss
        loss_DC = 0.5 * (loss_creal + loss_cfake)

        loss_DC.backward()
        optimizer_DC.step()
        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [Ds loss: %f] [Dc loss: %f][G loss: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_DS.item(),
                loss_DC.item(),
                loss_G.item(),
                time_left,
            )
        )
        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator_s.state_dict(),
                    "saved_models/%s/discriminators_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator_c.state_dict(),
                    "saved_models/%s/discriminatorc_%d.pth" % (opt.dataset_name, epoch))


#[Epoch 48/200] [Batch 921/1334] [Ds loss: 0.095980] [Dc loss: 0.000278][G loss: 1.156492] ETA: 3:09:01.304492.4931431
