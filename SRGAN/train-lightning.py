import pytorch_lightning as pl
from torch import nn
from torch import optim
from utils import load_checkpoint, save_checkpoint, plot_examples
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from dataset import MyImageFolder
import config
import torch

class SRGAN(pl.LightningModule):
    def __init__(self, in_channels=3, img_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        self.gen = Generator(in_channels=in_channels)
        self.disc = Discriminator(in_channels=in_channels, features=features)
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.vgg_loss = VGGLoss()
        self.automatic_optimization = False 

    def forward(self, x):
        return self.gen(x)

    def training_step(self, batch, batch_idx):
        opt_disc, opt_gen = self.optimizers()

        device = next(self.parameters()).device  # Get the device of the model
        low_res, high_res = batch[0].to(device), batch[1].to(device)  # Move tensors to the same device as the model
        fake = self.gen(low_res)

        # Train Discriminator
        opt_disc.zero_grad()
        disc_real = self.disc(high_res)
        disc_fake = self.disc(fake.detach())
        disc_loss_real = self.bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        )
        disc_loss_fake = self.bce(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = disc_loss_fake + disc_loss_real
        self.log('loss_disc', loss_disc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        loss_disc.backward()
        opt_disc.step()

        # Train Generator
        opt_gen.zero_grad()
        disc_fake = self.disc(fake)
        adversarial_loss = 1e-3 * self.bce(disc_fake, torch.ones_like(disc_fake))
        loss_for_vgg = 0.006 * self.vgg_loss(fake, high_res)
        gen_loss = loss_for_vgg + adversarial_loss
        self.log('gen_loss', gen_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        gen_loss.backward()
        opt_gen.step()

        return {'loss_disc': loss_disc, 'gen_loss': gen_loss}

    def configure_optimizers(self):
        opt_gen = optim.Adam(self.gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
        opt_disc = optim.Adam(self.disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
        return [opt_disc, opt_gen], []

def main():
    dataset = MyImageFolder(root_dir="../celeba/img_align_celeba")
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
    model = SRGAN()
    trainer = pl.Trainer(max_epochs=config.NUM_EPOCHS)
    trainer.fit(model, loader)

if __name__ == "__main__":
    main()
