import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Discriminator, Generator, DCGAN

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
NUM_EPOCHS = 15
FEATURES_DISC = 64
FEATURES_GEN = 64

if __name__ == "__main__":
    model = DCGAN(learning_rate=LEARNING_RATE)
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
    trainer.fit(model)
