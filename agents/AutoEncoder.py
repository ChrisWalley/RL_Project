import numpy as np
import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(fc2_input_dim[2], 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=1),
            nn.Conv2d(64, 1, kernel_size=3, stride=1),
            nn.ReLU()
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        example_convolution = self.encoder_cnn(torch.zeros(1, fc2_input_dim[2], fc2_input_dim[0], fc2_input_dim[1]))
        self.resultant_shape = example_convolution.size()[1:]
        self.resultant_size = int(np.prod(self.resultant_shape))

        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(self.resultant_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, encoded_space_dim)
        )

    def forward(self, x):
        x = torch.tensor(x).to(next(self.encoder_cnn.parameters()).device).unsqueeze(0)
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3844),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(dim=2,
                                      unflattened_size=(62, 62))

        self.decoder_conv = nn.Sequential(

            nn.ConvTranspose2d(1, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, fc2_input_dim[2], kernel_size=3, stride=2, output_padding=1)
        )

        self.Upsample = nn.Upsample(size=(fc2_input_dim[0], fc2_input_dim[1]))

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = self.Upsample(x)
        x = torch.sigmoid(x)
        return x