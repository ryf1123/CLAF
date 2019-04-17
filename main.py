import os
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

# special use libs
from pypianoroll import Multitrack, Track

# our libs
from claf.Generator import Generator
from claf.Discriminator import Discriminator
from claf.Encoder import Encoder

import models.dd as dd
import models.ew as ew
import models.hybrid as hybrid

from utils import *

# set some PARAMETERS
cuda = True if torch.cuda.is_available() else False
batch_size = 32
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def arg_parse():
    # parse the parameters, this will be helpful in the release version, but may have no use now
    parser = argparse.ArgumentParser(description='CLAF: ...')

    parser.add_argument("--model", default='hybrid',
                        choices=['dd', 'ew', 'hybrid'])
    parser.add_argument("--pretrained", action='store_true')

    return parser.parse_args()


class Model():
    def __init__(self, n_tracks=4, beat_resolution=12):
        self.gen = Generator()
        self.dis = Discriminator()
        self.enc = Encoder()

        if cuda:
            self.gen = self.gen.cuda()
            self.dis = self.dis.cuda()
            self.enc = self.enc.cuda()

        self.start_epoch = 0

        def optimizer(param):
            return torch.optim.Adam(param, lr=0.0002, betas=(0.5, 0.999))

        self.optimizer_E = optimizer(self.enc.parameters())
        self.optimizer_G = optimizer(self.gen.parameters())
        self.optimizer_D = optimizer(self.dis.parameters())

    def load(self):
        if 'saved_model' in os.listdir('.'):
            print("[info ] restoring model and parameters")
            state = torch.load('./saved_model')

            self.enc.load_state_dict(state['state_dict_enc'])
            self.gen.load_state_dict(state['state_dict_gen'])
            self.dis.load_state_dict(state['state_dict_dis'])

            self.optimizer_E.load_state_dict(state['optimizer_E'])
            self.optimizer_G.load_state_dict(state['optimizer_G'])
            self.optimizer_D.load_state_dict(state['optimizer_D'])

            self.start_epoch = state['epoch']

            print("[info ] Found history done")

        else:
            print("[info ] Found history fail, No history")


def main():
    args = arg_parse()

    data = load_data_from_npz()
    model = Model()

    if args.pretrained:
        model.load()

    if args.model == 'hybrid':
        hybrid.train(model, data)
    elif args.model == 'ew':
        ew.train(model, data)
    elif args.model == 'dd':
        dd.train(model, data)
    else:
        pass


if __name__ == '__main__':
    main()
