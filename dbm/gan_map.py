import torch
from torch.optim import Adam, RMSprop
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from dbm.util import make_grid_np, rand_rot_mtx, rot_mtx_batch, voxelize_gauss, make_dir, avg_blob, voxelize_gauss_batch
from dbm.torch_energy import *
from dbm.output import *
from dbm.recurrent_generator import Recurrent_Generator
from dbm.mol_generator import Mol_Generator
from dbm.mol_generator_AA import Mol_Generator_AA
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

        g = Mol_Generator(data, train=train, rand_rot=False)

        self.elems = g.all_elems()

        self.resolution = cfg.getint('grid', 'resolution')
        self.delta_s = cfg.getfloat('grid', 'length') / cfg.getint('grid', 'resolution')
        self.sigma_aa = cfg.getfloat('grid', 'sigma_aa')
        self.sigma_cg = cfg.getfloat('grid', 'sigma_cg')

        if cfg.getboolean('training', 'rand_rot'):
            self.rand_rot = True
            print("using random rotations during training...")
        else:
            self.rand_rot = False
        self.align = int(cfg.getboolean('universe', 'align'))

        self.cg_env = cfg.getboolean('model', 'cg_env')

        self.grid = make_grid_np(self.delta_s, self.resolution)


    def __len__(self):
        return len(self.elems)

    def __getitem__(self, ndx):
        if self.rand_rot and self.train:
            R = rand_rot_mtx(self.data.align)
        else:
            R = np.eye(3, dtype=np.float32)

        d = self.elems[ndx]


        aa_coords_intra = np.dot(d['aa_positions_intra'], R.T)
        aa_blobbs_intra = voxelize_gauss(aa_coords_intra, self.sigma_aa, self.grid)
        aa_features_intra = d['aa_intra_featvec'][:, :, None, None, None] * aa_blobbs_intra[:, None, :, :, :]

        gen_input = np.sum(aa_features_intra, 0)
        #aa_features_intra = aa_blobbs_intra

        cg_positions_intra = voxelize_gauss(np.dot(d['cg_positions_intra'], R.T), self.sigma_cg, self.grid)
        target = cg_positions_intra

        aa_coords = aa_coords_intra
        cg_coords_inter = np.zeros((1,3))
        #if d['aa_positions_inter']:

        aa_coords_inter = np.dot(d['aa_positions_inter'], R.T)
        aa_blobbs_inter = voxelize_gauss(aa_coords_inter, self.sigma_aa, self.grid)
        aa_features_inter = d['aa_inter_featvec'][:, :, None, None, None] * aa_blobbs_inter[:, None, :, :, :]
        aa_features_inter = np.sum(aa_features_inter, 0)
        gen_input = np.concatenate((gen_input, aa_features_inter), 0)

        aa_coords = np.concatenate((aa_coords_intra, aa_coords_inter), 0)


        cg_coords_inter = np.dot(d['cg_positions_inter'], R.T)
        cg_blobbs_inter = voxelize_gauss(cg_coords_inter, self.sigma_cg, self.grid)
        cg_features_inter = d['cg_inter_featvec'][:, :, None, None, None] * cg_blobbs_inter[:, None, :, :, :]
        cg_features_inter = np.sum(cg_features_inter, 0)
        crit_input_real = np.concatenate((target, cg_features_inter), 0)



        #print(target)
        #print(target.shape)
        #print(features.shape)

        energy_ndx_aa = (d['aa_bond_ndx'], d['aa_ang_ndx'], d['aa_dih_ndx'], d['aa_lj_intra_ndx'], d['aa_lj_ndx'])
        energy_ndx_cg = (d['cg_bond_ndx'], d['cg_ang_ndx'], d['cg_dih_ndx'], d['cg_lj_intra_ndx'],  d['cg_lj_ndx'])

        """
        #print([a.type.name for a in d['aa_mol'].atoms])
        fig = plt.figure(figsize=(20, 20))
        n_chns = 4
        colours = ['red', 'black', 'green', 'blue']
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # ax.scatter(mol_aa.com[0], mol_aa.com[1],mol_aa.com[2], s=20, marker='o', color='blue', alpha=0.5)

        for i in range(0, self.resolution):
            for j in range(0, self.resolution):
                for k in range(0, self.resolution):
                    for n in range(0,1):
                        #ax.scatter(i,j,k, s=2, marker='o', color='black', alpha=min(target[n,i,j,k], 1.0))
                        if features[n,i,j,k] > 0.1:
                            ax.scatter(i,j,k, s=2, marker='o', color='black', alpha=min(features[n,i,j,k], 1.0))
        #carb_ndx = [1,4,7,10,13,16,19,22]
        carb_ndx = [0,1,2,3]
        print(aa_coords_intra)
        for z in range(0, len(aa_coords_intra)):
            if z in carb_ndx:
                ax.scatter(aa_coords_intra[z, 0]/self.delta_s + self.resolution/2-0.5, aa_coords_intra[z, 1]/self.delta_s+ self.resolution/2-0.5, aa_coords_intra[z, 2]/self.delta_s+ self.resolution/2-0.5, s=8, marker='o', color='red')
            #else:
            #    ax.scatter(aa_coords_intra[z, 0]/self.delta_s + self.resolution/2, aa_coords_intra[z, 1]/self.delta_s+ self.resolution/2, aa_coords_intra[z, 2]/self.delta_s+ self.resolution/2, s=4, marker='o', color='blue')
        """
        """
        f = d['aa_positions_intra']
        carb_ndx = [1,4,7,10,13,16,19,22]
        for z in range(0, len(f)):
            if z in carb_ndx:
                ax.scatter(f[z, 0], f[z, 1], f[z, 2], s=4, marker='o', color='red')
            else:
                ax.scatter(f[z, 0], f[z, 1], f[z, 2], s=4, marker='o', color='blue')
        



        ax.set_xlim3d(1.0, self.resolution)
        ax.set_ylim3d(1.0, self.resolution)
        ax.set_zlim3d(1.0, self.resolution)
        #ax.set_xticks(np.arange(-1, 1, step=0.5))
        #ax.set_yticks(np.arange(-1, 1, step=0.5))
        #ax.set_zticks(np.arange(-1, 1, step=0.5))
        #ax.tick_params(labelsize=6)
        #plt.plot([0.0, 0.0], [0.0, 0.0], [-1.0, 1.0])
        plt.show()
        """


        #print("features", features.dtype)
        #print("target", target.dtype)
        #print("aa_coords_intra", aa_coords_intra.dtype)
        #print("aa_coords", aa_coords.dtype)

        elems = (gen_input, crit_input_real, aa_features_inter, aa_coords_intra, aa_coords, cg_coords_inter)

        return elems, energy_ndx_aa, energy_ndx_cg


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
        self.ff_aa = self.data.ff_aa
        self.ff_cg = self.data.ff_cg

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
        self.cg_env = cfg.getboolean('model', 'cg_env')
        self.n_env_mols = cfg.getint('universe', 'n_env_mols')

        self.feature_dim = self.ff_aa.n_atom_chns
        self.feature_dim += self.ff_aa.n_atom_chns
        """
        if self.n_env_mols != 0:
            self.feature_dim += self.ff_aa.n_atom_chns
            if self.cg_env:
                self.feature_dim += self.ff_cg.n_atom_chns
        """
        self.target_dim = self.ff_cg.n_atoms
        self.critic_dim = self.ff_aa.n_atom_chns + self.target_dim
        #if self.cond:
        #self.critic_dim += self.feature_dim

        self.z_dim = int(cfg.getint('model', 'noise_dim'))

        #print(self.ff_aa.n_atom_chns, self.n_input, self.n_out, self.ff_cg.n_atoms)

        self.step = 0
        self.epoch = 0

        # Make Dirs for saving
        self.out = OutputHandler(
            self.name,
            self.cfg.getint('training', 'n_checkpoints'),
            self.cfg.get('model', 'output_dir'),
        )
        self.energy_aa = Energy_torch(self.ff_aa, self.device)
        self.energy_cg = Energy_torch(self.ff_cg, self.device)

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

    def overlap_loss(self, aa_mol, cg_mol):
        aa_mol = torch.flatten(torch.sum(aa_mol, 1), 1)
        aa_mol = aa_mol / torch.sum(aa_mol, 1, keepdim=True)
        cg_mol = torch.flatten(torch.sum(cg_mol, 1), 1)
        cg_mol = cg_mol / torch.sum(cg_mol, 1, keepdim=True)

        overlap_loss = (aa_mol * (aa_mol / cg_mol).log()).sum(1)
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

    def get_energies_aa(self, aa_coords_intra, aa_coords, energy_ndx):

        bond_ndx, angle_ndx, dih_ndx, lj_intra_ndx, lj_ndx = energy_ndx
        if bond_ndx.size()[1]:
            b_energy = self.energy_aa.bond(aa_coords_intra, bond_ndx)
        else:
            b_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if angle_ndx.size()[1]:
            a_energy = self.energy_aa.angle(aa_coords_intra, angle_ndx)
        else:
            a_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if dih_ndx.size()[1]:
            d_energy = self.energy_aa.dih(aa_coords_intra, dih_ndx)
        else:
            d_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if lj_ndx.size()[1]:
            l_energy = self.energy_aa.lj(aa_coords, lj_ndx)
        else:
            l_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        #print(l_energy)
        return torch.mean(b_energy), torch.mean(a_energy), torch.mean(d_energy), torch.mean(l_energy)

    def get_energies_cg(self, atom_grid, coords_inter, energy_ndx):
        coords = avg_blob(
            atom_grid,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma_cg'),
            device=self.device,
        )

        bond_ndx, angle_ndx, dih_ndx, lj_intra_ndx, lj_ndx = energy_ndx
        if bond_ndx.size()[1]:
            b_energy = self.energy_cg.bond(coords, bond_ndx)
        else:
            b_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if angle_ndx.size()[1]:
            a_energy = self.energy_cg.angle(coords, angle_ndx)
        else:
            a_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if dih_ndx.size()[1]:
            d_energy = self.energy_cg.dih(coords, dih_ndx)
        else:
            d_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        if self.cg_env and self.n_env_mols:
            coords = torch.cat((coords, coords_inter), 1)
            l_energy = self.energy_cg.lj(coords, lj_ndx)
        elif lj_intra_ndx.size()[1]:
            l_energy = self.energy_cg.lj(coords, lj_intra_ndx)
        else:
            l_energy = torch.zeros([], dtype=torch.float32, device=self.device)
        return torch.mean(b_energy), torch.mean(a_energy), torch.mean(d_energy), torch.mean(l_energy)

    def detach(self, t):
        t = tuple([c.detach().cpu().numpy() for c in t])
        return t

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
                elems, energy_ndx_aa, energy_ndx_cg = train_batch

                if n == n_critic:
                    for key, value in c_loss_dict.items():
                        self.out.add_scalar(key, value, global_step=self.step)
                    g_loss_dict = self.train_step_gen(elems, energy_ndx_aa, energy_ndx_cg)
                    for key, value in g_loss_dict.items():
                        self.out.add_scalar(key, value, global_step=self.step)
                    #print(c_loss_dict)
                    tqdm_train_iterator.set_description('D: {:.2f}, G: {:.2f}, E_cg: {:.2f}, {:.2f}, {:.2f}, {:.2f}, E_aa: {:.2f}, {:.2f}, {:.2f}, {:.2f}, OL: {:.2f}'.format(c_loss_dict['Critic/wasserstein'],
                                                                                   g_loss_dict['Generator/wasserstein'],
                                                                                   g_loss_dict['Generator/e_bond_cg'],
                                                                                   g_loss_dict['Generator/e_angle_cg'],
                                                                                   g_loss_dict['Generator/e_dih_cg'],
                                                                                   g_loss_dict['Generator/e_lj_cg'],
                                                                                   g_loss_dict['Generator/e_bond_aa'],
                                                                                   g_loss_dict['Generator/e_angle_aa'],
                                                                                   g_loss_dict['Generator/e_dih_aa'],
                                                                                   g_loss_dict['Generator/e_lj_aa'],
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
                        elems, energy_ndx_aa, energy_ndx_cg = val_batch
                        g_loss_dict = self.train_step_gen(elems, energy_ndx_aa, energy_ndx_cg, backprop=False)
                        for key, value in g_loss_dict.items():
                            self.out.add_scalar(key, value, global_step=self.step, mode='val')
                    self.step += 1
                    n = 0

                else:
                    c_loss_dict = self.train_step_critic(elems)
                    #print("l", c_loss)
                    n += 1

            #tqdm.write(g_loss_dict)
            """
            tqdm.write('epoch {} steps {} : D: {} G: {}, E_cg: {}, {}, {}, {}, E_aa: {}, {}, {}, {}, OL: {}'.format(
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

    def val(self):
        start = timer()

        resolution = self.cfg.getint('grid', 'resolution')
        grid_length = self.cfg.getfloat('grid', 'length')
        delta_s = self.cfg.getfloat('grid', 'length') / self.cfg.getint('grid', 'resolution')
        sigma_aa = self.cfg.getfloat('grid', 'sigma_aa')
        sigma_cg = self.cfg.getfloat('grid', 'sigma_cg')
        grid = torch.from_numpy(make_grid_np(delta_s, resolution)).to(self.device)

        cg_env = self.cfg.getboolean('model', 'cg_env')
        val_bs = self.cfg.getint('validate', 'batchsize')

        samples_aa = self.data.samples_val_aa
        pos_dict = {}
        for sample in samples_aa:
            for a in sample.atoms:
                pos_dict[a] = a.pos

        g = Mol_Generator_AA(self.data, train=False, rand_rot=False)
        all_elems = list(g)


        try:
            self.generator.eval()
            self.critic.eval()

            for o in range(0, self.cfg.getint('validate', 'n_gibbs')):
                for ndx in range(0, len(all_elems), val_bs):
                    with torch.no_grad():
                        batch = all_elems[ndx:min(ndx + val_bs, len(all_elems))]

                        aa_positions_intra = np.array([d['aa_positions_intra'] for d in batch])
                        aa_intra_featvec = np.array([d['aa_intra_featvec'] for d in batch])

                        aa_positions_intra = torch.from_numpy(aa_positions_intra).to(self.device).float()
                        aa_blobbs_intra = self.to_voxel(aa_positions_intra, grid, sigma_aa)

                        features = torch.from_numpy(aa_intra_featvec[:, :, :, None, None, None]).to(self.device) * aa_blobbs_intra[:, :, None, :, :, :]
                        features = torch.sum(features, 1)


                        aa_positions_inter = np.array([d['aa_positions_inter'] for d in batch])
                        aa_inter_featvec = np.array([d['aa_inter_featvec'] for d in batch])

                        aa_positions_inter = torch.from_numpy(aa_positions_inter).to(self.device).float()
                        aa_blobbs_inter = self.to_voxel(aa_positions_inter, grid, sigma_aa)

                        features_aa_inter = torch.from_numpy(aa_inter_featvec[:, :, :, None, None, None]).to(self.device) * aa_blobbs_inter[:, :, None, :, :, :]
                        features_aa_inter = torch.sum(features_aa_inter, 1)


                        gen_input = torch.cat((features, features_aa_inter), 1)

                        """
                        cg_positions_inter = np.array([d['cg_positions_inter'] for d in batch])
                        cg_inter_featvec = np.array([d['cg_inter_featvec'] for d in batch])

                        cg_positions_inter = torch.from_numpy(cg_positions_inter).to(self.device).float()
                        cg_blobbs_inter = self.to_voxel(cg_positions_inter, grid, sigma_aa)

                        features_cg_inter = torch.from_numpy(cg_inter_featvec[:, :, :, None, None, None]).to(self.device) * cg_blobbs_inter[:, :, None, :, :, :]
                        features_cg_inter = torch.sum(features_cg_inter, 1)

                        #features = torch.cat((features, features_cg_inter), 1)

                        gen_input = torch.cat((features, features_cg_inter), 1)
                        """

                        mols = np.array([d['aa_mol'] for d in batch])


                        #elems, energy_ndx_aa, energy_ndx_cg = val_batch
                        #features, _, aa_coords_intra, aa_coords = elems
                        if self.z_dim != 0:
                            z = torch.empty(
                                [features.shape[0], self.z_dim],
                                dtype=torch.float32,
                                device=self.device,
                            ).normal_()

                            fake_mol = self.generator(z, gen_input)
                        else:
                            fake_mol = self.generator(gen_input)

                        coords = avg_blob(
                            fake_mol,
                            res=resolution,
                            width=grid_length,
                            sigma=sigma_cg,
                            device=self.device,)
                        for positions, mol in zip(coords, mols):
                            positions = positions.detach().cpu().numpy()
                            positions = np.dot(positions, mol.rot_mat.T)
                            for pos, atom in zip(positions, mol.atoms):
                                atom.pos = pos + mol.com

            samples_dir = self.out.output_dir / "samples"
            samples_dir.mkdir(exist_ok=True)

            for sample in self.data.samples_val_aa:
                #sample.write_gro_file(samples_dir / (sample.name + str(self.step) + ".gro"))
                sample.write_aa_gro_file(samples_dir / (sample.name + ".gro"))
                for a in sample.atoms:
                    a.pos = pos_dict[a]
                    #pos_dict[a] = a.pos
                #sample.kick_beads()
        finally:
            self.generator.train()
            self.critic.train()
            print("validation took ", timer()-start, "secs")


    def train_step_critic(self, elems):

        gen_input, crit_input_real, aa_features_inter, _, _, _ = elems

        #features, target, _, _, _ = elems

        #c_loss = torch.zeros([], dtype=torch.float32, device=self.device)

        if self.z_dim != 0:
            z = torch.empty(
                [gen_input.shape[0], self.z_dim],
                dtype=torch.float32,
                device=self.device,
            ).normal_()

            fake_mol = self.generator(z, gen_input)
        else:
            fake_mol = self.generator(gen_input)


        """
        fake_mol2 = fake_mol.detach().cpu().numpy()
        fig = plt.figure(figsize=(20, 20))
        n_chns = 4
        colours = ['red', 'black', 'green', 'blue']
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # ax.scatter(mol_aa.com[0], mol_aa.com[1],mol_aa.com[2], s=20, marker='o', color='blue', alpha=0.5)
        for i in range(0, 8):
            for j in range(0, 8):
                for k in range(0, 8):
                    for n in range(0,1):
                        #ax.scatter(i,j,k, s=2, marker='o', color='black', alpha=min(target[n,i,j,k], 1.0))
                        ax.scatter(i,j,k, s=2, marker='o', color='black', alpha=fake_mol2[0, n,i,j,k])

            #ax.set_xlim3d(-1.0, 1.0)
            #ax.set_ylim3d(-1.0, 1.0)
            #ax.set_zlim3d(-1.0, 1.0)
            #ax.set_xticks(np.arange(-1, 1, step=0.5))
            #ax.set_yticks(np.arange(-1, 1, step=0.5))
            #ax.set_zticks(np.arange(-1, 1, step=0.5))
            #ax.tick_params(labelsize=6)
            #plt.plot([0.0, 0.0], [0.0, 0.0], [-1.0, 1.0])
        plt.show()
        """

        #fake_data = torch.cat([fake_atom, features], dim=1)
        #real_data = torch.cat([target_atom[:, None, :, :, :], features], dim=1)

        #critic
        """
        if self.cond:
            fake_data = torch.cat([fake_mol, features], dim=1)
            real_data = torch.cat([target, features], dim=1)
        else:
            fake_data = fake_mol
            real_data = target
        """

        real_data = crit_input_real
        fake_data = torch.cat([fake_mol, aa_features_inter], dim=1)



        critic_fake = self.critic(fake_data)
        critic_real = self.critic(real_data)


        #loss
        c_wass = self.critic_loss(critic_real, critic_fake)
        c_eps = self.epsilon_penalty(1e-3, critic_real)
        c_loss = c_wass + c_eps

        c_gp = 10.0 * self.gradient_penalty(real_data, fake_data)
        if self.use_gp:
            c_loss += c_gp

        #print(c_loss)
        #print("::::::::::")
        self.opt_critic.zero_grad()
        c_loss.backward()
        self.opt_critic.step()

        c_loss_dict = {"Critic/wasserstein": c_wass.detach().cpu().numpy(),
                       "Critic/eps": c_eps.detach().cpu().numpy(),
                       "Critic/gp": c_gp.detach().cpu().numpy(),
                       "Critic/total": c_loss.detach().cpu().numpy()
                       }

        return c_loss_dict


    def train_step_gen(self, elems, energy_ndx_aa, energy_ndx_cg, backprop=True):

        gen_input, _, aa_features_inter, aa_coords_intra, aa_coords, cg_coords_inter = elems

        g_loss = torch.zeros([], dtype=torch.float32, device=self.device)

        if self.z_dim != 0:
            z = torch.empty(
                [gen_input.shape[0], self.z_dim],
                dtype=torch.float32,
                device=self.device,
            ).normal_()

            fake_mol = self.generator(z, gen_input)
        else:
            fake_mol = self.generator(gen_input)

        """
        if self.cond:
            fake_data = torch.cat([fake_mol, features], dim=1)
        else:
            fake_data = fake_mol
        """

        fake_data = torch.cat([fake_mol, aa_features_inter], dim=1)


        critic_fake = self.critic(fake_data)

        #loss
        g_wass = self.generator_loss(critic_fake)
        #print("g_wass", g_wass)
        g_overlap = self.overlap_loss(gen_input[:, :self.ff_aa.n_atom_chns], fake_mol)
        if self.use_ol:
            g_loss += g_wass + self.ol_weight * g_overlap
        else:
            g_loss += g_wass

        #g_loss = g_overlap

        #real_atom_grid = torch.where(repl[:, :, None, None, None], atom_grid, target_atom[:, None, :, :, :])
        #fake_atom_grid = torch.where(repl[:, :, None, None, None], atom_grid, fake_atom)

        e_bond_cg, e_angle_cg, e_dih_cg, e_lj_cg = self.get_energies_cg(fake_mol, cg_coords_inter, energy_ndx_cg)
        e_bond_aa, e_angle_aa, e_dih_aa, e_lj_aa = self.get_energies_aa(aa_coords_intra, aa_coords, energy_ndx_aa)

        if self.use_energy:
            if self.prior_mode == 'match':

                e_bond_cg_target, e_angle_cg_target, e_dih_cg_target, e_lj_cg_target = self.get_energies_cg(target, cg_coords_inter, energy_ndx_cg)

                #print("target")
                #print(e_bond_cg_target, e_angle_cg_target, e_dih_cg_target, e_lj_cg_target)
                #print("gen")
                #print(e_bond_cg, e_angle_cg, e_dih_cg, e_lj_cg)
                b_loss = torch.mean(torch.abs(e_bond_cg_target - e_bond_cg))
                a_loss = torch.mean(torch.abs(e_angle_cg_target - e_angle_cg))
                d_loss = torch.mean(torch.abs(e_dih_cg_target - e_dih_cg))
                l_loss = torch.mean(torch.abs(e_lj_cg_target - e_lj_cg))
                g_loss += self.energy_weight() * (b_loss + a_loss + d_loss + l_loss)
            elif self.prior_mode == 'min':
                g_loss += self.energy_weight() * (e_bond_cg + e_angle_cg + e_dih_cg + e_lj_cg)



        #g_loss = g_wass + self.prior_weight() * energy_loss
        #g_loss = g_wass

        if backprop:
            self.opt_generator.zero_grad()
            g_loss.backward()
            #for param in self.generator.parameters():
            #    print(param.grad)
            self.opt_generator.step()


        g_loss_dict = {"Generator/wasserstein": g_wass.detach().cpu().numpy(),
                       "Generator/e_bond_cg": e_bond_cg.detach().cpu().numpy(),
                       "Generator/e_angle_cg": e_angle_cg.detach().cpu().numpy(),
                       "Generator/e_dih_cg": e_dih_cg.detach().cpu().numpy(),
                       "Generator/e_lj_cg": e_lj_cg.detach().cpu().numpy(),
                       "Generator/e_bond_aa": e_bond_aa.detach().cpu().numpy(),
                       "Generator/e_angle_aa": e_angle_aa.detach().cpu().numpy(),
                       "Generator/e_dih_aa": e_dih_aa.detach().cpu().numpy(),
                       "Generator/e_lj_aa": e_lj_aa.detach().cpu().numpy(),
                       "Generator/overlap": g_overlap.detach().cpu().numpy()}

        return g_loss_dict


    def to_voxel(self, coords, grid, sigma):
        coords = coords[..., None, None, None]
        return torch.exp(-1.0 * torch.sum((grid - coords) * (grid - coords), axis=2) / sigma).float()


