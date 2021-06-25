import torch
from torch.optim import Adam, RMSprop
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from dbm.util import make_grid_np, rand_rot_mtx, rot_mtx_batch, voxelize_gauss, make_dir, avg_blob, voxelize_gauss_batch
from dbm.torch_energy import *
from dbm.output import *
from dbm.recurrent_generator import Recurrent_Generator
from dbm.mol_rec_generator import Mol_Rec_Generator
#from dbm.mol_generator_AA import Mol_Generator_AA
from tqdm import tqdm
import numpy as np
#from tqdm import tqdm
from timeit import default_timer as timer
import os
import math
#from configparser import ConfigParser
#import mdtraj as md
#from universe import *
import dbm.model as model
from dbm.data import *
from dbm.stats import *
from scipy import constants
#import dbm.tf_utils as tf_utils
#import dbm.tf_energy as tf_energy
from copy import deepcopy
from shutil import copyfile
from contextlib import redirect_stdout
from operator import add
from itertools import cycle
import gc

#tf.compat.v1.disable_eager_execution()

torch.set_default_dtype(torch.float32)
torch.set_printoptions(edgeitems=50)

class DS(Dataset):
    def __init__(self, data, cfg, train=True):

        self.data = data
        self.train = train
        self.n_env_mols = int(cfg.getint('universe', 'n_env_mols'))

        g = Mol_Rec_Generator(data, train=train, rand_rot=False)

        self.elems = g.all_elems()

        self.resolution = cfg.getint('grid', 'resolution')
        self.delta_s = cfg.getfloat('grid', 'length') / cfg.getint('grid', 'resolution')

        self.sigma = cfg.getfloat('grid', 'sigma_out')

        if cfg.getboolean('training', 'rand_rot'):
            self.rand_rot = True
            print("using random rotations during training...")
        else:
            self.rand_rot = False
        self.align = int(cfg.getboolean('universe', 'align'))

        self.out_env = cfg.getboolean('model', 'out_env')

        self.grid = make_grid_np(self.delta_s, self.resolution)


    def __len__(self):
        return len(self.elems)

    def __getitem__(self, ndx):
        if self.rand_rot and self.train:
            R = rand_rot_mtx(self.data.align)
        else:
            R = np.eye(3, dtype=np.float32)

        d = self.elems[ndx]


        targets = voxelize_gauss(np.dot(d['targets'], R.T), self.sigma, self.grid)
        atom_grid = voxelize_gauss(np.dot(d['positions'], R.T), self.sigma, self.grid)

        elems = (targets, d['featvec'], d['repl'])
        initial = (atom_grid)
        energy_ndx = (d['bond_ndx'], d['angle_ndx'], d['dih_ndx'], d['lj_ndx'])

        #print(d['ljs_ndx'].shape)
        #energy_ndx = (bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx)

        #return atom_grid, bead_grid, target_atom, target_type, aa_feat, repl, mask, energy_ndx
        #return atom_grid, cg_features, target_atom, d['target_type'], d['aa_feat'], d['repl'], d['mask'], energy_ndx, d['aa_pos']
        return elems, initial, energy_ndx


class GAN():

    def __init__(self, device, cfg):

        self.device = device
        self.cfg = cfg

        self.bs = self.cfg.getint('training', 'batchsize')

        #Data pipeline
        self.data = Data(cfg, save=False)
        ds_train = DS(self.data, cfg)
        if len(ds_train) != 0:
            self.loader_train = DataLoader(
                ds_train,
                batch_size=self.bs,
                shuffle=True,
                drop_last=False,
                pin_memory=True,
                num_workers=0,
            )
        else:
            self.loader_train = []
        self.steps_per_epoch = int(len(self.loader_train) / (self.cfg.getint('training', 'n_critic') + 1))
        print(len(self.loader_train), self.steps_per_epoch)
        self.ff = self.data.ff_inp

        if len(ds_train) != 0:
            ds_val = DS(self.data, cfg, train=False)
            if len(ds_val) != 0:
                self.loader_val = DataLoader(
                    ds_val,
                    batch_size=self.bs,
                    shuffle=True,
                    drop_last=False,
                    pin_memory=True,
                    num_workers=0,
                )
            else:
                self.loader_val = []

        #model
        self.name = cfg.get('model', 'name')

        self.cond = cfg.getboolean('model', 'cond')
        if self.cond and not self.data.pairs:
            raise Exception('conditional GAN can only be used with pairs of snapshots for both resolutions.')
        self.out_env = cfg.getboolean('model', 'out_env')
        self.n_env_mols = cfg.getint('universe', 'n_env_mols')

        self.feature_dim = self.ff.n_channels
        """
        if self.n_env_mols != 0:
            self.feature_dim += self.ff_inp.n_atom_chns
            if self.out_env:
                self.feature_dim += self.ff_out.n_atom_chns
        """
        self.target_dim = 1
        self.critic_dim = self.feature_dim + self.target_dim
        #if self.cond:
        #self.critic_dim += self.feature_dim

        self.z_dim = int(cfg.getint('model', 'noise_dim'))

        #print(self.ff_inp.n_atom_chns, self.n_input, self.n_out, self.ff_out.n_atoms)

        self.step = 0
        self.epoch = 0

        # Make Dirs for saving
        self.out = OutputHandler(
            self.name,
            self.cfg.getint('training', 'n_checkpoints'),
            self.cfg.get('model', 'output_dir'),
        )
        self.energy = Energy_torch(self.ff, self.device)
        #self.energy_out = Energy_torch(self.ff_out, self.device)

        self.gauss_hist_bond = GaussianHistogram_Dis(bins=64, min=0.0, max=0.8, sigma=0.005, ff=self.ff, device=device)
        self.gauss_hist_angle = GaussianHistogram_Angle(bins=64, min=0, max=180, sigma=2.0, ff=self.ff, device=device)
        self.gauss_hist_dih = GaussianHistogram_Dih(bins=64, min=0, max=180, sigma=4.0, ff=self.ff, device=device)
        self.gauss_hist_nb = GaussianHistogram_Dis(bins=64, min=0.0, max=2.0, sigma=0.02, ff=self.ff, device=device)

        self.ol_weight = cfg.getfloat('prior', 'ol')

        prior_weights = self.cfg.get('prior', 'weights')
        self.prior_weights = [float(v) for v in prior_weights.split(",")]
        prior_schedule = self.cfg.get('prior', 'schedule')
        try:
            self.prior_schedule = np.array([0] + [int(v) for v in prior_schedule.split(",")])
        except:
            self.prior_schedule = [0]

        self.ratio_bonded_nonbonded = cfg.getfloat('prior', 'ratio_bonded_nonbonded')
        self.prior_mode = cfg.get('prior', 'mode')
        print(self.prior_mode)

        #Model selection
        #if cfg.get('model', 'model_type') == "tiny":
        #    print("Using tiny model")
        if self.z_dim != 0:
            self.generator = model.G_tiny_with_noise(z_dim=self.z_dim,
                                                n_input=self.feature_dim,
                                                n_output=self.target_dim,
                                                start_channels=self.cfg.getint('model', 'n_chns'),
                                                fac=1,
                                                sn=self.cfg.getint('model', 'sn_gen'),
                                                device=device)
            print("Using tiny generator with noise")
        else:
            self.generator = model.G_tiny(n_input=self.feature_dim,
                                            n_output=self.target_dim,
                                            start_channels=self.cfg.getint('model', 'n_chns'),
                                            fac=1,
                                            sn=self.cfg.getint('model', 'sn_gen'),
                                            device=device)
            print("Using tiny generator without noise")

        if cfg.getint('grid', 'resolution') == 8:
            self.critic = model.C_tiny_mbd(in_channels=self.critic_dim,
                                              start_channels=self.cfg.getint('model', 'n_chns'),
                                              fac=1, sn=self.cfg.getint('model', 'sn_crit'),
                                              device=device)
            print("Using tiny critic with resolution 8")


        else:
            self.critic = model.C_tiny16(in_channels=self.critic_dim,
                                              start_channels=self.cfg.getint('model', 'n_chns'),
                                              fac=1, sn=self.cfg.getint('model', 'sn_crit'),
                                              device=device)

            print("Using tiny critic with resolution 16")


        self.use_gp = cfg.getboolean('model', 'gp')
        self.use_ol = cfg.getboolean('model', 'ol')
        self.use_energy = cfg.getboolean('model', 'energy')




        self.critic.to(device=device)
        self.generator.to(device=device)
        #self.mse.to(device=device)

        lr_gen = cfg.getfloat('training', 'lr_gen')
        lr_crit = cfg.getfloat('training', 'lr_crit')
        #self.opt_generator_pretrain = Adam(self.generator.parameters(), lr=lr_gen, betas=(0, 0.9))
        self.opt_generator = Adam(self.generator.parameters(), lr=lr_gen, betas=(0, 0.9))
        self.opt_critic = Adam(self.critic.parameters(), lr=lr_crit, betas=(0, 0.9))

        self.restored_model = False
        self.restore_latest_checkpoint()

    def energy_weight(self):
        try:
            ndx = next(x[0] for x in enumerate(self.prior_schedule) if x[1] > self.epoch) - 1
        except:
            ndx = len(self.prior_schedule) - 1
        if ndx > 0 and self.prior_schedule[ndx] == self.epoch:
            weight = self.prior_weights[ndx-1] + self.prior_weights[ndx] * (self.step - self.epoch*self.steps_per_epoch) / self.steps_per_epoch
        else:
            weight = self.prior_weights[ndx]

        return weight


    def make_checkpoint(self):
        return self.out.make_checkpoint(
            self.step,
            {
                "generator": self.generator.state_dict(),
                "critic": self.critic.state_dict(),
                "opt_generator": self.opt_generator.state_dict(),
                "opt_critic": self.opt_critic.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
            },
        )

    def restore_latest_checkpoint(self):
        latest_ckpt = self.out.latest_checkpoint()
        if latest_ckpt is not None:
            checkpoint = torch.load(latest_ckpt)
            self.generator.load_state_dict(checkpoint["generator"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.opt_generator.load_state_dict(checkpoint["opt_generator"])
            self.opt_critic.load_state_dict(checkpoint["opt_critic"])
            self.step = checkpoint["step"]
            self.epoch = checkpoint["epoch"]
            self.restored_model = True
            print("restored model!!!")

        self.out.prune_checkpoints()

    def map_to_device(self, tup):
        return tuple(tuple(y.to(device=self.device) for y in x) if type(x) is list else x.to(device=self.device) for x in tup)

    def generator_loss(self, critic_fake):
        return (-1.0 * critic_fake).mean()

    def overlap_loss(self, inp_mol, out_mol):
        inp_mol = torch.flatten(torch.sum(inp_mol, 1), 1)
        inp_mol = inp_mol / torch.sum(inp_mol, 1, keepdim=True)
        out_mol = torch.flatten(torch.sum(out_mol, 1), 1)
        out_mol = out_mol / torch.sum(out_mol, 1, keepdim=True)

        overlap_loss = (inp_mol * (inp_mol / out_mol).log()).sum(1)
        return torch.mean(overlap_loss)

    def critic_loss(self, critic_real, critic_fake):
        loss_on_generated = critic_fake.mean()
        loss_on_real = critic_real.mean()

        loss = loss_on_generated - loss_on_real
        return loss

    def epsilon_penalty(self, epsilon, critic_real_outputs):
        if epsilon > 0:
            penalties = torch.pow(critic_real_outputs, 2)
            penalty = epsilon * penalties.mean()
            return penalty
        return 0.0

    def gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, 1, device=self.device)
        alpha = alpha.expand_as(real_data)

        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated.to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.critic(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size(), device=self.device),
                               create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)

        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradients_norm = ((gradients_norm - 1) ** 2)
        #gradients_norm = gradients_norm * mask

        # Return gradient penalty
        return gradients_norm.mean()

    def get_energies_inp(self, inp_coords_intra, inp_coords, energy_ndx):

        bond_ndx, angle_ndx, dih_ndx, lj_intra_ndx, lj_ndx = energy_ndx
        if bond_ndx.size()[1]:
            b_energy = self.energy_inp.bond(inp_coords_intra, bond_ndx)
        else:
            b_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if angle_ndx.size()[1]:
            a_energy = self.energy_inp.angle(inp_coords_intra, angle_ndx)
        else:
            a_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if dih_ndx.size()[1]:
            d_energy = self.energy_inp.dih(inp_coords_intra, dih_ndx)
        else:
            d_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if lj_ndx.size()[1]:
            l_energy = self.energy_inp.lj(inp_coords, lj_ndx)
        else:
            l_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        #print(l_energy)
        return torch.mean(b_energy), torch.mean(a_energy), torch.mean(d_energy), torch.mean(l_energy)

    def get_energies(self, atom_grid, energy_ndx):
        coords = avg_blob(
            atom_grid,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma_out'),
            device=self.device,
        )

        bond_ndx, angle_ndx, dih_ndx, lj_ndx = energy_ndx
        if bond_ndx.size()[1]:
            b_energy = self.energy.bond(coords, bond_ndx)
        else:
            b_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if angle_ndx.size()[1]:
            a_energy = self.energy.angle(coords, angle_ndx)
        else:
            a_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if dih_ndx.size()[1]:
            d_energy = self.energy.dih(coords, dih_ndx)
        else:
            d_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if lj_ndx.size()[1]:
            l_energy = self.energy.lj(coords, lj_ndx)
        else:
            l_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        return torch.mean(b_energy), torch.mean(a_energy), torch.mean(d_energy), torch.mean(l_energy)


    def dstr_loss(self, real_atom_grid, fake_atom_grid, energy_ndx):
        real_coords = avg_blob(
            real_atom_grid,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma_out'),
            device=self.device,
        )
        fake_coords = avg_blob(
            fake_atom_grid,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma_out'),
            device=self.device,
        )
        bond_ndx, angle_ndx, dih_ndx, lj_ndx = energy_ndx

        if bond_ndx.size()[1] and bond_ndx.size()[1]:
            #b_dstr_real = self.gauss_hist_bond(self.energy.bond_dstr(real_coords, bond_ndx))
            #b_dstr_fake = self.gauss_hist_bond(self.energy.bond_dstr(fake_coords, bond_ndx))
            b_dstr_inp = self.gauss_hist_bond(real_coords, bond_ndx)
            b_dstr_out = self.gauss_hist_bond(fake_coords, bond_ndx)
            b_dstr_avg = 0.5 * (b_dstr_inp + b_dstr_out)

            b_dstr_loss = 0.5 * ((b_dstr_inp * (b_dstr_inp / b_dstr_avg).log()).sum(0) + (b_dstr_out * (b_dstr_out / b_dstr_avg).log()).sum(0))

            if self.step % 50 == 0:
                fig = plt.figure()
                ax = plt.gca()
                x = [h * 0.4/64 for h in range(0,64)]
                ax.plot(x, b_dstr_inp.detach().cpu().numpy()[:,0], label='inp')
                ax.plot(x, b_dstr_out.detach().cpu().numpy()[:,0], label='out')
                #ax.plot(x, a_dstr_avg.detach().cpu().numpy()[:,0], label='avg')
                #ax.text(0.1, 0.1, "JSD: "+str(a_dstr_loss.detach().cpu().numpy()))
                self.out.add_fig("bond", fig, global_step=self.step)
                plt.close(fig)


        else:
            b_dstr_loss = torch.zeros([], dtype=torch.float32, device=self.device)
        if angle_ndx.size()[1] and angle_ndx.size()[1]:
            a_dstr_inp = self.gauss_hist_angle(real_coords, angle_ndx)
            a_dstr_out = self.gauss_hist_angle(fake_coords, angle_ndx)
            a_dstr_avg = 0.5 * (a_dstr_inp + a_dstr_out)

            a_dstr_loss = 0.5 * ((a_dstr_inp * (a_dstr_inp / a_dstr_avg).log()).sum(0) + (a_dstr_out * (a_dstr_out / a_dstr_avg).log()).sum(0))

            if self.step % 50 == 0:
                fig = plt.figure()
                ax = plt.gca()
                x = [h * 180.0/64 for h in range(0,64)]
                ax.plot(x, a_dstr_inp.detach().cpu().numpy()[:,0], label='inp')
                ax.plot(x, a_dstr_out.detach().cpu().numpy()[:,0], label='out')
                #ax.plot(x, a_dstr_avg.detach().cpu().numpy()[:,0], label='avg')
                #ax.text(0.1, 0.1, "JSD: "+str(a_dstr_loss.detach().cpu().numpy()))
                self.out.add_fig("angle", fig, global_step=self.step)
                plt.close(fig)


            #print(a_dstr_loss)
            #print(a_dstr_loss.size())

        else:
            a_dstr_loss = torch.zeros([], dtype=torch.float32, device=self.device)
        if dih_ndx.size()[1] and dih_ndx.size()[1]:
            d_dstr_inp = self.gauss_hist_bond(real_coords, dih_ndx)
            d_dstr_out = self.gauss_hist_bond(fake_coords, dih_ndx)
            d_dstr_avg = 0.5 * (d_dstr_inp + d_dstr_out)

            d_dstr_loss = 0.5 * ((d_dstr_inp * (d_dstr_inp / d_dstr_avg).log()).sum(0) + (d_dstr_out * (d_dstr_out / d_dstr_avg).log()).sum(0))

            if self.step % 50 == 0:
                fig = plt.figure()
                ax = plt.gca()
                x = [h * 180.0/64 for h in range(0,64)]
                ax.plot(x, d_dstr_inp.detach().cpu().numpy()[:,0], label='ref')
                ax.plot(x, d_dstr_out.detach().cpu().numpy()[:,0], label='fake')
                #ax.plot(x, a_dstr_avg.detach().cpu().numpy()[:,0], label='avg')
                #ax.text(0.1, 0.1, "JSD: "+str(a_dstr_loss.detach().cpu().numpy()))
                self.out.add_fig("dih", fig, global_step=self.step)
                plt.close(fig)

            #print(d_dstr_loss)
            #print(d_dstr_loss.size())
        else:
            d_dstr_loss = torch.zeros([], dtype=torch.float32, device=self.device)


        if lj_ndx.size()[1] and lj_ndx.size()[1]:
            nb_dstr_inp = self.gauss_hist_nb(real_coords, lj_ndx)
            nb_dstr_out = self.gauss_hist_nb(fake_coords, lj_ndx)
            nb_dstr_avg = 0.5 * (nb_dstr_inp + nb_dstr_out)

            nb_dstr_loss = 0.5 * ((nb_dstr_inp * (nb_dstr_inp / nb_dstr_avg).log()).sum(0) + (nb_dstr_out * (nb_dstr_out / nb_dstr_avg).log()).sum(0))

            if self.step % 50 == 0:
                fig = plt.figure()
                ax = plt.gca()
                x = [h * 2.0/64 for h in range(0,64)]
                ax.plot(x, nb_dstr_inp.detach().cpu().numpy()[:,0], label='inp')
                ax.plot(x, nb_dstr_out.detach().cpu().numpy()[:,0], label='out')
                #ax.plot(x, a_dstr_avg.detach().cpu().numpy()[:,0], label='avg')
                #ax.text(0.1, 0.1, "JSD: "+str(a_dstr_loss.detach().cpu().numpy()))
                self.out.add_fig("nonbonded", fig, global_step=self.step)
                plt.close(fig)

            #print(b_dstr_loss)
            #print(b_dstr_loss.size())
        else:
            nb_dstr_loss = torch.zeros([], dtype=torch.float32, device=self.device)
        #print(torch.sum(b_dstr_loss))
        #print(torch.sum(a_dstr_loss))
        #print(torch.sum(d_dstr_loss))
        #print(torch.sum(nb_dstr_loss))

        return torch.sum(b_dstr_loss), torch.sum(a_dstr_loss), torch.sum(d_dstr_loss), torch.sum(nb_dstr_loss)


    def detach(self, t):
        t = tuple([c.detach().cpu().numpy() for c in t])
        return t

    def transpose_and_zip(self, args):
        args = tuple(torch.transpose(x, 0, 1) for x in args)
        elems = zip(*args)
        return elems

    def train(self):
        steps_per_epoch = len(self.loader_train)
        n_critic = self.cfg.getint('training', 'n_critic')
        n_save = int(self.cfg.getint('training', 'n_save'))
        
        epochs = tqdm(range(self.epoch, self.cfg.getint('training', 'n_epoch')))
        epochs.set_description('epoch: ')
        for epoch in epochs:
            n = 0
            #loss_epoch = [[]]*11
            val_iterator = iter(self.loader_val)
            tqdm_train_iterator = tqdm(self.loader_train, total=steps_per_epoch, leave=False)
            for train_batch in tqdm_train_iterator:

                train_batch = self.map_to_device(train_batch)
                elems, initial, energy_ndx  = train_batch
                elems = self.transpose_and_zip(elems)

                if n == n_critic:
                    for key, value in c_loss_dict.items():
                        self.out.add_scalar(key, value, global_step=self.step)
                    g_loss_dict = self.train_step_gen(elems, initial, energy_ndx)
                    for key, value in g_loss_dict.items():
                        self.out.add_scalar(key, value, global_step=self.step)
                    #print(c_loss_dict)
                    tqdm_train_iterator.set_description('D: {:.2f}, G: {:.2f}, E_out: {:.2f}, {:.2f}, {:.2f}, {:.2f}, E_inp: {:.2f}, {:.2f}, {:.2f}, {:.2f}, OL: {:.2f}'.format(c_loss_dict['Critic/wasserstein'],
                                                                                   g_loss_dict['Generator/wasserstein'],
                                                                                   g_loss_dict['Generator/e_bond_out'],
                                                                                   g_loss_dict['Generator/e_angle_out'],
                                                                                   g_loss_dict['Generator/e_dih_out'],
                                                                                   g_loss_dict['Generator/e_lj_out'],
                                                                                   g_loss_dict['Generator/e_bond_inp'],
                                                                                   g_loss_dict['Generator/e_angle_inp'],
                                                                                   g_loss_dict['Generator/e_dih_inp'],
                                                                                   g_loss_dict['Generator/e_lj_inp'],
                                                                                   g_loss_dict['Generator/overlap']))

                    #for value, l in zip([c_loss] + list(g_loss_dict.values()), loss_epoch):
                    #    l.append(value)

                    if self.loader_val:
                        try:
                            val_batch = next(val_iterator)
                        except StopIteration:
                            val_iterator = iter(self.loader_val)
                            val_batch = next(val_iterator)
                        val_batch = self.map_to_device(val_batch)
                        elems, initial, energy_ndx  = val_batch
                        elems = self.transpose_and_zip(elems)
                        g_loss_dict = self.train_step_gen(elems, initial, energy_ndx, backprop=False)
                        for key, value in g_loss_dict.items():
                            self.out.add_scalar(key, value, global_step=self.step, mode='val')
                    self.step += 1
                    n = 0

                else:
                    c_loss_dict = self.train_step_critic(elems, initial)
                    #print("l", c_loss)
                    n += 1

            #tqdm.write(g_loss_dict)
            """
            tqdm.write('epoch {} steps {} : D: {} G: {}, E_out: {}, {}, {}, {}, E_inp: {}, {}, {}, {}, OL: {}'.format(
                self.epoch,
                self.step,
                sum(loss_epoch[0]) / len(loss_epoch[0]),
                sum(loss_epoch[1]) / len(loss_epoch[1]),
                sum(loss_epoch[2]) / len(loss_epoch[2]),
                sum(loss_epoch[3]) / len(loss_epoch[3]),
                sum(loss_epoch[4]) / len(loss_epoch[4]),
                sum(loss_epoch[5]) / len(loss_epoch[5]),
                sum(loss_epoch[6]) / len(loss_epoch[6]),
                sum(loss_epoch[7]) / len(loss_epoch[7]),
                sum(loss_epoch[8]) / len(loss_epoch[8]),
                sum(loss_epoch[9]) / len(loss_epoch[9]),
                sum(loss_epoch[10]) / len(loss_epoch[10]),

            ))
            """

            self.epoch += 1

            if self.epoch % n_save == 0:
                self.make_checkpoint()
                self.out.prune_checkpoints()
                self.val()

    def to_tensor_and_zip(self, *args):
        args = tuple(torch.from_numpy(x).float().to(self.device) if x.dtype == np.dtype(np.float64) else torch.from_numpy(x).to(self.device) for x in args)
        #args = tuple(torch.transpose(x, 0, 1) for x in args)
        elems = zip(*args)
        return elems

    def to_tensor(self, t):
        return tuple(torch.from_numpy(x).to(self.device) for x in t)

    def transpose(self, t):
        return tuple(torch.transpose(x, 0, 1) for x in t)

    def insert_dim(self, t):
        return tuple(x[None, :] for x in t)

    def repeat(self, t, bs):
        return tuple(torch.stack(bs*[x]) for x in t)

    def val(self):
        start = timer()

        resolution = self.cfg.getint('grid', 'resolution')
        grid_length = self.cfg.getfloat('grid', 'length')
        delta_s = self.cfg.getfloat('grid', 'length') / self.cfg.getint('grid', 'resolution')
        sigma_inp = self.cfg.getfloat('grid', 'sigma_inp')
        sigma_out = self.cfg.getfloat('grid', 'sigma_out')
        grid = torch.from_numpy(make_grid_np(delta_s, resolution)).to(self.device)



        out_env = self.cfg.getboolean('model', 'out_env')
        val_bs = self.cfg.getint('validate', 'batchsize')

        rot_mtxs = torch.from_numpy(rot_mtx_batch(val_bs)).to(self.device).float()
        rot_mtxs_transposed = torch.from_numpy(rot_mtx_batch(val_bs, transpose=True)).to(self.device).float()

        samples_inp = self.data.samples_val_inp
        pos_dict = {}
        for sample in samples_inp:
            for a in sample.atoms:
                pos_dict[a] = a.pos

        #generators = []
        #for n in range(0, self.cfg.getint('validate', 'n_gibbs')):
        #    generators.append(iter(Mol_Generator_AA(self.data, train=False, rand_rot=False)))
        #all_elems = list(g)


        try:
            self.generator.eval()
            self.critic.eval()

            for n in range(0, self.cfg.getint('validate', 'n_gibbs')):
                g = iter(Mol_Rec_Generator(self.data, train=False, rand_rot=False))
                for d in g:
                    with torch.no_grad():
                        #batch = all_elems[ndx:min(ndx + val_bs, len(all_elems))]

                        inp_positions = np.array([d['positions']])
                        #inp_featvec = np.array([d['inp_intra_featvec']])

                        inp_positions = torch.matmul(torch.from_numpy(inp_positions).to(self.device).float(), rot_mtxs)
                        aa_grid = self.to_voxel(inp_positions, grid, sigma_inp)

                        #features = torch.from_numpy(inp_featvec[:, :, :, None, None, None]).to(self.device) * inp_blobbs[:, :, None, :, :, :]
                        #features = torch.sum(features, 1)

                        mol = d['mol']

                        elems = (d['featvec'], d['repl'])
                        elems = self.transpose(self.insert_dim(self.to_tensor(elems)))

                        energy_ndx = (d['bond_ndx'], d['angle_ndx'], d['dih_ndx'], d['lj_ndx'])
                        energy_ndx = self.repeat(self.to_tensor(energy_ndx), val_bs)

                        generated_atoms = []
                        for featvec, repl in zip(*elems):
                            features = torch.sum(aa_grid[:, :, None, :, :, :] * featvec[:, :, :, None, None, None], 1)

                            # generate fake atom
                            if self.z_dim != 0:
                                z = torch.empty(
                                    [features.shape[0], self.z_dim],
                                    dtype=torch.float32,
                                    device=self.device,
                                ).normal_()

                                fake_atom = self.generator(z, features)
                            else:
                                fake_atom = self.generator(features)
                            generated_atoms.append(fake_atom)

                            # update aa grids
                            aa_grid = torch.where(repl[:, :, None, None, None], aa_grid, fake_atom)

                        # generated_atoms = torch.stack(generated_atoms, dim=1)
                        generated_atoms = torch.cat(generated_atoms, dim=1)

                        coords = avg_blob(
                            generated_atoms,
                            res=self.cfg.getint('grid', 'resolution'),
                            width=self.cfg.getfloat('grid', 'length'),
                            sigma=self.cfg.getfloat('grid', 'sigma_out'),
                            device=self.device,
                        )

                        coords = torch.matmul(coords, rot_mtxs_transposed)
                        coords = torch.sum(coords, 0) / val_bs

                        #for positions, mol in zip(coords, mols):
                        positions = coords.detach().cpu().numpy()
                        positions = np.dot(positions, mol.rot_mat.T)
                        for pos, atom in zip(positions, mol.atoms):
                            atom.pos = pos + mol.com

                samples_dir = self.out.output_dir / "samples"
                samples_dir.mkdir(exist_ok=True)

                for sample in self.data.samples_val_inp:
                    #sample.write_gro_file(samples_dir / (sample.name + str(self.step) + ".gro"))
                    sample.write_aa_gro_file(samples_dir / (sample.name + "_" +str(n) + ".gro"))
                    for a in sample.atoms:
                        a.pos = pos_dict[a]
                        #pos_dict[a] = a.pos
                #sample.kick_beads()
        finally:
            self.generator.train()
            self.critic.train()
            print("validation took ", timer()-start, "secs")



    def train_step_critic(self, elems, initial):
        c_loss = torch.zeros([], dtype=torch.float32, device=self.device)

        fake_atom_grid = initial.clone()
        real_atom_grid = initial.clone()


        for target_atom, featvec, repl in elems:
            #prepare input for generator
            z = torch.empty(
                [target_atom.shape[0], self.z_dim],
                dtype=torch.float32,
                device=self.device,
            ).normal_()

            features_real = torch.sum(real_atom_grid[:, :, None, :, :, :] * featvec[:, :, :, None, None, None], 1)
            features_fake = torch.sum(fake_atom_grid[:, :, None, :, :, :] * featvec[:, :, :, None, None, None], 1)

            #generate fake atom
            if self.z_dim != 0:
                z = torch.empty(
                    [features_fake.shape[0], self.z_dim],
                    dtype=torch.float32,
                    device=self.device,
                ).normal_()

                fake_atom = self.generator(z, features_fake)
            else:
                fake_atom = self.generator(features_fake)

            fake_data = torch.cat([fake_atom, features_fake], dim=1)
            real_data = torch.cat([target_atom[:, None, :, :, :], features_real], dim=1)

            #critic
            critic_fake = self.critic(fake_data)
            critic_real = self.critic(real_data)

            #mask
            critic_fake = torch.squeeze(critic_fake)
            critic_real = torch.squeeze(critic_real)

            #loss
            c_wass = self.critic_loss(critic_real, critic_fake)
            c_eps = self.epsilon_penalty(1e-3, critic_real)
            c_loss += c_wass + c_eps
            if self.use_gp:
                c_gp = self.gradient_penalty(real_data, fake_data)
                c_loss += c_gp

            #update aa grids
            fake_atom_grid = torch.where(repl[:,:,None,None,None], fake_atom_grid, fake_atom)
            real_atom_grid = torch.where(repl[:,:,None,None,None], real_atom_grid, target_atom[:, None, :, :, :])

        self.opt_critic.zero_grad()
        c_loss.backward()
        self.opt_critic.step()

        c_loss_dict = {"Critic/wasserstein": c_wass.detach().cpu().numpy(),
                       "Critic/eps": c_eps.detach().cpu().numpy(),
                       "Critic/gp": c_gp.detach().cpu().numpy(),
                       "Critic/total": c_loss.detach().cpu().numpy()
                       }

        return c_loss_dict


    def train_step_gen(self, elems, initial, energy_ndx, backprop=True):

        g_loss = torch.zeros([], dtype=torch.float32, device=self.device)

        fake_atom_grid = initial.clone()
        real_atom_grid = initial.clone()

        fake_mol, real_mol = [], []

        for target_atom, featvec, repl in elems:
            #prepare input for generator
            features = torch.sum(fake_atom_grid[:, :, None, :, :, :] * featvec[:, :, :, None, None, None], 1)

            #generate fake atom
            if self.z_dim != 0:
                z = torch.empty(
                    [features.shape[0], self.z_dim],
                    dtype=torch.float32,
                    device=self.device,
                ).normal_()

                fake_atom = self.generator(z, features)
            else:
                fake_atom = self.generator(features)

            #critic
            critic_fake = self.critic(torch.cat([fake_atom, features], dim=1))

            #loss
            g_loss += self.generator_loss(critic_fake)
            #g_loss += g_wass


            #update aa grids
            fake_atom_grid = torch.where(repl[:,:,None,None,None], fake_atom_grid, fake_atom)
            real_atom_grid = torch.where(repl[:,:,None,None,None], real_atom_grid, target_atom[:, None, :, :, :])

            fake_mol.append(fake_atom)
            real_mol.append(target_atom[:, None, :, :, :])


        e_bond_out, e_angle_out, e_dih_out, e_lj_out = self.get_energies(fake_atom_grid, energy_ndx)
        e_bond_inp, e_angle_inp, e_dih_inp, e_lj_inp = self.get_energies(real_atom_grid, energy_ndx)

        b_dstr, a_dstr, d_dstr, nb_dstr = self.dstr_loss(real_atom_grid, fake_atom_grid, energy_ndx)

        fake_mol = torch.cat(fake_mol, dim=1)
        real_mol = torch.cat(real_mol, dim=1)

        g_overlap = self.overlap_loss(real_mol, fake_mol)


        if self.use_energy:
            if self.prior_mode == 'match':

                b_loss = torch.mean(torch.abs(e_bond_inp - e_bond_out))
                a_loss = torch.mean(torch.abs(e_angle_inp - e_angle_out))
                d_loss = torch.mean(torch.abs(e_dih_inp - e_dih_out))
                l_loss = torch.mean(torch.abs(e_lj_inp - e_lj_out))
                g_loss = g_loss + self.energy_weight() * ((b_loss + a_loss + d_loss)* self.ratio_bonded_nonbonded + l_loss)
            elif self.prior_mode == 'min':
                g_loss = g_loss + self.energy_weight() * ((e_bond_out + e_angle_out + e_dih_out)*self.ratio_bonded_nonbonded + e_lj_out)
            elif self.prior_mode == "dstr":
                g_loss = g_loss + self.energy_weight() * ((b_dstr + a_dstr + d_dstr)*self.ratio_bonded_nonbonded + nb_dstr)



        #g_loss = g_wass + self.prior_weight() * energy_loss
        #g_loss = g_wass

        if backprop:
            self.opt_generator.zero_grad()
            g_loss.backward()
            #for param in self.generator.parameters():
            #    print(param.grad)
            self.opt_generator.step()


        g_loss_dict = {"Generator/wasserstein": g_loss.detach().cpu().numpy(),
                       "Generator/e_bond_out": e_bond_out.detach().cpu().numpy(),
                       "Generator/e_angle_out": e_angle_out.detach().cpu().numpy(),
                       "Generator/e_dih_out": e_dih_out.detach().cpu().numpy(),
                       "Generator/e_lj_out": e_lj_out.detach().cpu().numpy(),
                       "Generator/e_bond_inp": e_bond_inp.detach().cpu().numpy(),
                       "Generator/e_angle_inp": e_angle_inp.detach().cpu().numpy(),
                       "Generator/e_dih_inp": e_dih_inp.detach().cpu().numpy(),
                       "Generator/e_lj_inp": e_lj_inp.detach().cpu().numpy(),
                       "Generator/overlap": g_overlap.detach().cpu().numpy()}

        return g_loss_dict


    def to_voxel(self, coords, grid, sigma):
        coords = coords[..., None, None, None]
        return torch.exp(-1.0 * torch.sum((grid - coords) * (grid - coords), axis=2) / sigma).float()


