import pypianoroll
import numpy as np
import torch
import os
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def load_data_from_npz():
    import SharedArray as sa
    return sa.attach("train_x_lpd_5_phr")


def calc_gp(netD, real_data, fake_data, batches_done):
    BATCH_SIZE = 64
    eps_x = torch.rand(BATCH_SIZE, 1)
    eps_x = eps_x.expand(BATCH_SIZE, int(
        real_data.nelement()/BATCH_SIZE)).contiguous()
    eps_x = eps_x.view(real_data.shape).cuda()
    fake_data = fake_data.view(real_data.shape)

    inter_x = eps_x * real_data + (1.0 - eps_x) * fake_data  # nodes['fake_x']
    dis_x_inter_out = netD(inter_x)

    gradient_x = torch.autograd.grad(
        outputs=dis_x_inter_out, inputs=inter_x,
        grad_outputs=torch.ones(
            dis_x_inter_out.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradient_x.view(gradient_x.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

    # print("[ *** ] gradient_penalty: ", gradient_penalty)

    return gradient_penalty


def get_real_data(train_real_data, batch_size):
    return torch.autograd.Variable(Tensor(train_real_data[(
        np.random.rand(batch_size) * train_real_data.shape[0]).astype(int), :]))


def save_music(musics, epoch):
    multitrack = []
    programs = [0, 0, 25, 33, 48]
    padding = np.zeros((192, 128), dtype=np.uint8)
    x = []
    os.makedirs('Gen_numpy', exist_ok=True)
    np.save("./Gen_numpy/%d_org.npy" % epoch, musics.cpu().detach().numpy())

    for i in range(len(musics[0])):
        roll_all = np.zeros((0, 128), dtype=np.uint8)
        for music in musics:
            roll = music[i]
            roll = np.reshape(roll.cpu().detach().numpy(), (192, 84))
            roll = np.pad(roll, ((0, 0), (24, 20)), 'constant')
            roll = (roll > 0.5) * 1
            roll_all = np.concatenate((roll_all, roll), 0)
            roll_all = np.concatenate((roll_all, padding), 0)

        x.append(roll_all)

        track = pypianoroll.Track(pianoroll=roll_all*100,
                                  program=programs[i], is_drum=i == 0, name='')
        multitrack.append(track)

    multitrack = pypianoroll.Multitrack(tracks=multitrack, tempo=120.0, downbeat=[
        0, 48, 96, 144], beat_resolution=12)
    os.makedirs('Gen_midi', exist_ok=True)
    multitrack.write('./Gen_midi/'+str(epoch)+'.mid')
