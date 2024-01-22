import torch
import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    
# Define the convolutional autoencoder class
class ConvAutoencoder(nn.Module):
    def __init__(self, config):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(9, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(512, config["hidden_size"], kernel_size=1),
            nn.AdaptiveMaxPool1d(config["dim_pool"]), 
            nn.Flatten()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (16,4,4,4)),
            nn.ConvTranspose3d(16, 4,kernel_size=2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose3d(4, 9,kernel_size=2, stride=2),
            nn.Flatten(2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
