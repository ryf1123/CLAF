import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.n_tracks = 5
        self.beat_resolution = 12
        self.batch_size = 32

        def conv_layer(i, o, k, s):
            return [
                nn.Conv3d(i, o, k, s),
                nn.LeakyReLU(0.1)
            ]

        self.pitch_time_layer = nn.Sequential(
            *conv_layer(5, 16, (1, 1, 12), (1, 1, 12)),
            *conv_layer(16, 32, (1, 3, 1), (1, 3, 1))
        )

        self.time_pitch_layer = nn.Sequential(
            *conv_layer(5, 16, (1, 3, 1), (1, 3, 1)),
            *conv_layer(16, 32, (1, 1, 12), (1, 1, 12))
        )

        self.merge_private_layer = nn.Sequential(
            *conv_layer(64, 64, 1, 1)
        )

        self.shared = nn.Sequential(
            *conv_layer(64, 128, (1, 4, 3), (1, 4, 2)),
            *conv_layer(128, 256, (1, 4, 3), (1, 4, 3))
        )

        self.chroma = nn.Sequential(
            *conv_layer(5, 32, (1, 1, 12), (1, 1, 12)),
            *conv_layer(32, 64, (1, 4, 1), (1, 4, 1))
        )

        self.on_off_set = nn.Sequential(
            *conv_layer(5, 16, (1, 3, 1), (1, 3, 1)),
            *conv_layer(16, 32, (1, 4, 1), (1, 4, 1)),
            *conv_layer(32, 64, (1, 4, 1), (1, 4, 1))
        )

        self.merge_all_streams = nn.Sequential(
            *conv_layer(384, 512, (2, 1, 1), (1, 1, 1))
        )

        self.dense = nn.Sequential(
            nn.Linear(512 * 3, 512)
        )

        self.x_to_mu = nn.Linear(512, 128)
        self.x_to_logvar = nn.Linear(512, 128)

    def _reparameterize(self, x):
        mu = self.x_to_mu(x)
        logvar = self.x_to_logvar(x)
        z = torch.randn(mu.size())
        z = z.cuda()
        z = mu + z * torch.exp(0.5 * logvar)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        return z, kld

    def forward(self, tensor_in):
        h = tensor_in
        n_beats = h.shape[3] // self.beat_resolution
        reshaped = tensor_in.reshape(
            -1, h.shape[1], h.shape[2], n_beats, self.beat_resolution, h.shape[4])

        summed = reshaped.sum(4)
        factor = int(h.shape[4]) // 12
        reshaped = summed[..., :(factor * 12)].reshape(
            -1, h.shape[1], h.shape[2], n_beats, factor, 12)
        chroma = reshaped.sum(4)

        padded = F.pad(tensor_in[:, :, :, :-1], (0, 0, 1, 0, 0, 0, 0, 0, 0, 0))
        on_off_set = (tensor_in - padded).sum(4, keepdim=True)

        s1 = self.pitch_time_layer(h)
        s2 = self.time_pitch_layer(h)
        h = torch.cat((s1, s2), 1)
        h = self.merge_private_layer(h)

        h = self.shared(h)
        c = self.chroma(chroma)
        o = self.on_off_set(on_off_set)
        h = torch.cat((h, c, o), 1)

        h = self.merge_all_streams(h)
        h = h.reshape(-1, self.batch_size, 3 * 512)
        h = self.dense(h)

        z, kld = self._reparameterize(h)
        z = z.reshape([self.batch_size, 128, 1, 1, 1])

        return z, kld
