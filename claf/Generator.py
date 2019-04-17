import numpy as np
import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.n_tracks = 5

        def tconv_layer(i, f, k, s):
            return [
                nn.ConvTranspose3d(i, f, k, s),
                nn.BatchNorm3d(f),
                nn.ReLU()
            ]

        self.model_1 = nn.Sequential(
            *tconv_layer(128, 512, (4, 1, 1), (4, 1, 1)),
            *tconv_layer(512, 256, (1, 4, 3), (1, 4, 3)),
            *tconv_layer(256, 128, (1, 4, 3), (1, 4, 2))
        )

        self.model_2_1 = nn.Sequential(
            *tconv_layer(128, 32, (1, 1, 12), (1, 1, 12)),
            *tconv_layer(32, 16, (1, 3, 1), (1, 3, 1))
        )
        self.model_2_2 = nn.Sequential(
            *tconv_layer(128, 32, (1, 1, 12), (1, 1, 12)),
            *tconv_layer(32, 16, (1, 3, 1), (1, 3, 1))
        )
        self.model_2_3 = nn.Sequential(
            *tconv_layer(128, 32, (1, 1, 12), (1, 1, 12)),
            *tconv_layer(32, 16, (1, 3, 1), (1, 3, 1))
        )
        self.model_2_4 = nn.Sequential(
            *tconv_layer(128, 32, (1, 1, 12), (1, 1, 12)),
            *tconv_layer(32, 16, (1, 3, 1), (1, 3, 1))
        )
        self.model_2_5 = nn.Sequential(
            *tconv_layer(128, 32, (1, 1, 12), (1, 1, 12)),
            *tconv_layer(32, 16, (1, 3, 1), (1, 3, 1))
        )

        self.model_3_1 = nn.Sequential(
            *tconv_layer(128, 32, (1, 3, 1), (1, 3, 1)),
            *tconv_layer(32, 16, (1, 1, 12), (1, 1, 12))
        )
        self.model_3_2 = nn.Sequential(
            *tconv_layer(128, 32, (1, 3, 1), (1, 3, 1)),
            *tconv_layer(32, 16, (1, 1, 12), (1, 1, 12))
        )
        self.model_3_3 = nn.Sequential(
            *tconv_layer(128, 32, (1, 3, 1), (1, 3, 1)),
            *tconv_layer(32, 16, (1, 1, 12), (1, 1, 12))
        )
        self.model_3_4 = nn.Sequential(
            *tconv_layer(128, 32, (1, 3, 1), (1, 3, 1)),
            *tconv_layer(32, 16, (1, 1, 12), (1, 1, 12))
        )
        self.model_3_5 = nn.Sequential(
            *tconv_layer(128, 32, (1, 3, 1), (1, 3, 1)),
            *tconv_layer(32, 16, (1, 1, 12), (1, 1, 12))
        )

        self.model_4_1 = nn.Sequential(
            nn.ConvTranspose3d(32, 1, (1, 1, 1), (1, 1, 1)),
            nn.BatchNorm3d(1)
        )
        self.model_4_2 = nn.Sequential(
            nn.ConvTranspose3d(32, 1, (1, 1, 1), (1, 1, 1)),
            nn.BatchNorm3d(1)
        )
        self.model_4_3 = nn.Sequential(
            nn.ConvTranspose3d(32, 1, (1, 1, 1), (1, 1, 1)),
            nn.BatchNorm3d(1)
        )
        self.model_4_4 = nn.Sequential(
            nn.ConvTranspose3d(32, 1, (1, 1, 1), (1, 1, 1)),
            nn.BatchNorm3d(1)
        )
        self.model_4_5 = nn.Sequential(
            nn.ConvTranspose3d(32, 1, (1, 1, 1), (1, 1, 1)),
            nn.BatchNorm3d(1)
        )

    def forward(self, z):
        model_1_out = self.model_1(z)
        model_2_out = [
            self.model_2_1(model_1_out),
            self.model_2_2(model_1_out),
            self.model_2_3(model_1_out),
            self.model_2_4(model_1_out),
            self.model_2_5(model_1_out)
        ]
        model_3_out = [
            self.model_3_1(model_1_out),
            self.model_3_2(model_1_out),
            self.model_3_3(model_1_out),
            self.model_3_4(model_1_out),
            self.model_3_5(model_1_out)
        ]
        model_23_out = [
            torch.cat((model_2_out[i], model_3_out[i]), 1)
            for i in range(self.n_tracks)
        ]
        model_4_out = [
            self.model_4_1(model_23_out[0]),
            self.model_4_2(model_23_out[1]),
            self.model_4_3(model_23_out[2]),
            self.model_4_4(model_23_out[3]),
            self.model_4_5(model_23_out[4])
        ]
        model_4_out = torch.cat(model_4_out, 1)

        return torch.tanh(model_4_out)
