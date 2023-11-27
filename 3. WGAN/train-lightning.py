import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from model import Discriminator, Generator, initialize_weights

BATCH_SIZE = 64

class GAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lr = 5e-5
        self.batch_size = 64
        self.image_size = 64
        self.channels_img = 3
        self.z_dim = 128
        self.features_critic = 64
        self.features_gen = 64
        self.critic_iterations = 5
        self.weight_clip = 0.01

        self.gen = Generator(self.z_dim, self.channels_img, self.features_gen)
        self.critic = Discriminator(self.channels_img, self.features_critic)
        initialize_weights(self.gen)
        initialize_weights(self.critic)

        self.automatic_optimization = False

    def forward(self, z):
        return self.gen(z)

    def training_step(self, batch):
        imgs, _ = batch
        batch_size = imgs.shape[0]
        noise = torch.randn(batch_size, self.z_dim, 1, 1).to(self.device)

        opt_critic, opt_gen = self.optimizers()  # Access optimizers like this

        # Train Critic
        for _ in range(self.critic_iterations):
            opt_critic.zero_grad()
            fake = self(noise)
            critic_real = self.critic(imgs).reshape(-1)
            critic_fake = self.critic(fake).reshape(-1)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            self.log('critic_loss', loss_critic, on_step=True)
            loss_critic.backward()
            opt_critic.step()

            # clip critic weights between -0.01, 0.01
            for p in self.critic.parameters():
                p.data.clamp_(-self.weight_clip, self.weight_clip)

        # Train Generator
        opt_gen.zero_grad()
        fake = self(noise)
        gen_fake = self.critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        self.log('gen_loss', loss_gen, on_step=True)
        loss_gen.backward()
        opt_gen.step()

        return {'loss': loss_gen, 'log': {'gen_loss': loss_gen, 'critic_loss': loss_critic}}

    def configure_optimizers(self):
        opt_critic = torch.optim.RMSprop(self.critic.parameters(), lr=self.lr)
        opt_gen = torch.optim.RMSprop(self.gen.parameters(), lr=self.lr)
        return [opt_critic, opt_gen]

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5 for _ in range(self.channels_img)], [0.5 for _ in range(self.channels_img)])
        ])
        dataset = datasets.ImageFolder(root="../celeba", transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

# Training
model = GAN()
trainer = pl.Trainer(max_epochs=20)
trainer.fit(model)
