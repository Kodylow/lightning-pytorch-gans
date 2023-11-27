import torchvision.datasets as datasets
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import gradient_penalty
from model import Discriminator, Generator, initialize_weights

class GAN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Hyperparameters
        self.lr = 1e-4
        self.z_dim = 100
        self.batch_size = 64
        self.image_size = 64
        self.features_gen = 64
        self.features_critic = 64
        self.channels_img = 3
        self.lambda_gp = 10
        self.critic_iterations = 5

        self.gen = Generator(self.z_dim, self.channels_img, self.features_gen)
        self.critic = Discriminator(self.channels_img, self.features_critic)
        initialize_weights(self.gen)
        initialize_weights(self.critic)

        self.device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fixed_noise = torch.randn(32, self.z_dim, 1, 1).to(self.device_type)

        self.automatic_optimization = False  # Set manual optimization

        self.current_step = 0

    def forward(self, z):
        return self.gen(z)

    def training_step(self, batch):
        opt_critic, opt_gen = self.optimizers()  # Access optimizers

        real, _ = batch
        cur_batch_size = real.shape[0]
        noise = torch.randn(cur_batch_size, self.z_dim, 1, 1, device=self.device_type)

        # Train Critic
        opt_critic.zero_grad()
        fake = self(noise)
        critic_real = self.critic(real).reshape(-1)
        critic_fake = self.critic(fake).reshape(-1)
        gp = gradient_penalty(self.critic, real, fake, device=self.device_type)
        loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.lambda_gp * gp
        self.log('critic_loss', loss_critic)
        loss_critic.backward()
        opt_critic.step()

        # Initialize loss_gen to None
        loss_gen = None

        # Train Generator
        if self.current_step % self.critic_iterations == 0:
            opt_gen.zero_grad()
            fake = self(noise)
            gen_fake = self.critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            self.log('gen_loss', loss_gen)
            loss_gen.backward()
            opt_gen.step()

        self.current_step += 1

        return loss_critic if self.current_step % self.critic_iterations == 0 else loss_gen

    def configure_optimizers(self):
        opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr, betas=(0.0, 0.9))
        opt_gen = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0.0, 0.9))
        return [opt_critic, opt_gen], []

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.CenterCrop(self.image_size),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5 for _ in range(self.channels_img)], [0.5 for _ in range(self.channels_img)])
        ])
        dataset = datasets.ImageFolder(root="../celeba", transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def on_train_epoch_end(self):
        with torch.no_grad():
            fake = self(self.fixed_noise)
            img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
            self.logger.experiment.add_image("Fake", img_grid_fake, self.current_epoch)

device = 'cuda'

model = GAN()
model.to(device)
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model)
