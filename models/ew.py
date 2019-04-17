from utils import *

import time
import torch
import numpy as np
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

batch_size = 32  # TOTALSIZE // epochs_num


def train(model, train_real_data, epochs_num=50000):
    # begin training
    for epoch in range(model.start_epoch, epochs_num):
        print()
        print("[data ] Epoch:%d/%d" % (epoch, epochs_num))

        t_begin = time.time()

        if epoch % 100 == 0 and epoch != 0:
            print("[state] begin saving history")
            # build parameter list to be saved later
            state = {
                'epoch': epoch,
                'state_dict_enc': model.enc.state_dict(),
                'state_dict_gen': model.gen.state_dict(),
                'state_dict_dis': model.dis.state_dict(),
                'optimizer_G': model.optimizer_G.state_dict(),
                'optimizer_D': model.optimizer_D.state_dict(),
                'optimizer_E': model.optimizer_E.state_dict(),
            }
            torch.save(state, 'saved_model')
            print("[Save history done]")

        # --------------------------------------
        #  Train Discriminator, Decoder by LGan
        # --------------------------------------

        for p in model.dis.parameters():
            p.requires_grad_(True)

        for _ in range(5):
            model.dis.zero_grad()
            model.gen.zero_grad()
            model.enc.zero_grad()

            batch_real_data = get_real_data(train_real_data, batch_size)

            z_real, _ = model.enc(batch_real_data)
            train_fake_data = model.gen(z_real)
            gen_loss_f = model.dis.forward(train_fake_data).mean()

            z = np.random.normal(size=[batch_size, 128, 1, 1, 1])
            train_noise_data = model.gen(Variable(Tensor(z)))
            gen_loss_p = model.dis.forward(train_noise_data).mean()
            gen_loss_r = model.dis.forward(batch_real_data).mean()

            print("[data ] gen_loss_f: ", gen_loss_f)
            print("[data ] gen_loss_p: ", gen_loss_p)
            print("[data ] gen_loss_r: ", gen_loss_r)

            gp_f = calc_gp(model.dis, batch_real_data, train_fake_data, epoch)
            gp_p = calc_gp(model.dis, batch_real_data, train_noise_data, epoch)
            gp = 0.5 * (gp_f + gp_p)

            dis_loss = 0.5 * (gen_loss_f + gen_loss_p) - gen_loss_r

            gp.backward(retain_graph=True)
            dis_loss.backward(retain_graph=True)
            model.optimizer_D.step()

            tmp_z = torch.tensor(-1.0).cuda()
            dis_loss.backward(tmp_z, retain_graph=True)
            dis_loss.backward(tmp_z)
            model.optimizer_G.step()

        # --------------------------------------------------
        #  Train Encoder, Generator, by Lprior, L dis
        # --------------------------------------------------

        for p in model.dis.parameters():
            p.requires_grad_(False)

        for _ in range(1):
            model.gen.zero_grad()

            batch_real_data = get_real_data(train_real_data, batch_size)
            z_real, kld = model.enc(batch_real_data)
            z_real.requires_grad_(True)

            train_fake_data = model.gen(z_real.detach())
            loss_element_wise = 0.01 * \
                (train_fake_data - batch_real_data).pow(2).sum() / batch_size

            if epoch and epoch % 100 == 0:
                save_music(train_fake_data, epoch)

            gen_loss = model.dis.forward(train_fake_data)
            gen_loss = - gen_loss.mean() + kld.mean() + loss_element_wise
            gen_loss.backward()

            model.optimizer_E.step()
            model.optimizer_G.step()

        t_end = time.time()

        print("[info ] duration of an epoch: %.4f" % (t_end - t_begin))
        print("[data ] gen_loss: %.4f" % (gen_loss))
        print("[data ] dis_loss: %.4f" % (dis_loss))
