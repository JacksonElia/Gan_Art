from data_loading import *
from discriminator import *
from generator import *
from gpu_manager import *

if __name__ == '__main__':
    batch_size = 128
    latent_size = 128
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    train_dl = load_data(batch_size, stats)
    train_dl, device = manage_gpu(train_dl)
    generator = Generator(latent_size).get_generator()
    discriminator = Discriminator().get_discriminator()
    discriminator = to_device(train_dl, discriminator)
    xb = torch.randn(batch_size, latent_size, 1, 1)  # random latent tensors
    fake_images = generator(xb)
    show_images(fake_images, stats)
