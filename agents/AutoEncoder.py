import numpy as np
import torch
from torch import nn

resultant_size = 0
class Encoder(nn.Module):

    def __init__(self, batch_size, encoded_space_dim, image_size):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(image_size[2], 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            torch.nn.MaxPool2d(2, stride=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            torch.nn.MaxPool2d(2, stride=1),
            nn.Conv2d(64, 1, kernel_size=3, stride=1),
            nn.ReLU()
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=0)

        example_convolution = self.encoder_cnn(torch.zeros(batch_size, image_size[2], image_size[0], image_size[1]))
        self.convolution_shape = example_convolution.size()

        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(int(np.prod(self.convolution_shape)), 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, encoded_space_dim)
        )

    def get_convolution_shape(self):
        return self.convolution_shape

    def forward(self, x):
        x = torch.nn.functional.normalize(x.to(next(self.encoder_cnn.parameters()).device))
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, image_size, convolution_shape):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 512),
            nn.Linear(512, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, int(np.prod(convolution_shape[1:]))),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=convolution_shape[1:])

        self.decoder_conv = nn.Sequential(

            nn.ConvTranspose2d(1, 64, kernel_size=3, stride=1),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),
            nn.ConvTranspose2d(64, image_size[2], kernel_size=3, stride=2, output_padding=1)
        )

        self.Upsample = nn.Upsample(size=(image_size[0], image_size[1]))

    def forward(self, x):
        x = torch.nn.functional.normalize(x.unsqueeze(0))
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = self.Upsample(x)
        x = torch.sigmoid(x)
        return x