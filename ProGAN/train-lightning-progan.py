import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from model import Discriminator, Generator
from utils import gradient_penalty
import torch
import config

class ProGAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.gen = Generator(
            config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
        )
        self.critic = Discriminator(
            config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
        )
        self.automatic_optimization = False
        self.alpha = 0.0
        self.steps = 0

    def forward(self, x):
        return self.gen(x, self.alpha, self.steps)

    def training_step(self, batch, batch_idx):
        real, _ = batch
        noise = torch.randn(real.shape[0], config.Z_DIM, 1, 1).to(self.device)
        opt_gen, opt_critic = self.optimizers()

        # Update alpha and steps based on current epoch
        self.alpha = min(1.0, self.alpha + 1.0 / config.PROGRESSIVE_EPOCHS[self.steps])
        if self.alpha == 1.0:
            self.alpha = 0.0
            self.steps += 1

        # training generator
        opt_gen.zero_grad()
        fake = self(noise)
        critic_fake = self.critic(fake.detach(), self.alpha, self.steps)
        loss_gen = -torch.mean(critic_fake)
        self.log('gen_loss', loss_gen, on_step=True)
        loss_gen.backward()
        opt_gen.step()

        # training critic
        opt_critic.zero_grad()
        critic_real = self.critic(real, self.alpha, self.steps)
        critic_fake = self.critic(fake, self.alpha, self.steps)
        gp = gradient_penalty(self.critic, real, fake, self.alpha, self.steps, device=self.device)
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake))
            + config.LAMBDA_GP * gp
            + (0.001 * torch.mean(critic_real ** 2))
        )
        self.log('critic_loss', loss_critic, on_step=True)
        loss_critic.backward()
        opt_critic.step()

        return loss_gen, loss_critic

    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(self.gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
        opt_critic = torch.optim.Adam(self.critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
        return [opt_gen, opt_critic]

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize(
                    [0.5 for _ in range(config.CHANNELS_IMG)],
                    [0.5 for _ in range(config.CHANNELS_IMG)],
                ),
            ]
        )
        dataset = ImageFolder(root=config.DATASET, transform=transform)
        return DataLoader(
            dataset,
            batch_size=config.BATCH_SIZES[self.current_epoch // sum(config.PROGRESSIVE_EPOCHS)],
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

if __name__ == "__main__":
    model = ProGAN()
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        filename='epoch-{epoch:02d}',
        monitor='val_loss',
        mode='min',
    )
    trainer = Trainer(
        max_epochs=sum(config.PROGRESSIVE_EPOCHS),
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)
