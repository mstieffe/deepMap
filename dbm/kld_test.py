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
        self.n_interatoms = int(cfg.getint('universe', 'n_inter_atoms'))

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
        aa_features_intra = np.sum(aa_features_intra, 0)
        #aa_features_intra = aa_blobbs_intra

        #print(np.dot(d['cg_positions_intra'], R.T))
        cg_positions_intra = voxelize_gauss(np.dot(d['cg_positions_intra'], R.T), self.sigma_cg, self.grid)

        #if d['aa_positions_inter']:
        if self.n_interatoms:
            aa_coords_inter = np.dot(d['aa_positions_inter'], R.T)
            aa_coords = np.concatenate((aa_coords_intra, aa_coords_inter), 0)
            aa_blobbs_inter = voxelize_gauss(aa_coords_inter, self.sigma_aa, self.grid)
            aa_features_inter = d['aa_inter_featvec'][:, :, None, None, None] * aa_blobbs_inter[:, None, :, :, :]
            aa_features_inter = np.sum(aa_features_inter, 0)
            features = np.concatenate((aa_features_intra, aa_features_inter), 0)

        else:
            features = aa_features_intra
            aa_coords = aa_coords_intra

        if self.n_interatoms:
            cg_positions_inter = voxelize_gauss(np.dot(d['cg_positions_inter'], R.T), self.sigma_cg, self.grid)

        target = cg_positions_intra

        energy_ndx_aa = (d['aa_bond_ndx'], d['aa_ang_ndx'], d['aa_dih_ndx'], d['aa_lj_intra_ndx'], d['aa_lj_ndx'])
        energy_ndx_cg = (d['cg_bond_ndx'], d['cg_ang_ndx'], d['cg_dih_ndx'], d['cg_lj_intra_ndx'],  d['cg_lj_ndx'])

        """
        fig = plt.figure(figsize=(20, 20))
        n_chns = 4
        colours = ['red', 'black', 'green', 'blue']
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # ax.scatter(mol_aa.com[0], mol_aa.com[1],mol_aa.com[2], s=20, marker='o', color='blue', alpha=0.5)
        for i in range(0, self.resolution):
            for j in range(0, self.resolution):
                for k in range(0, self.resolution):
                    for n in range(0,2):
                        #ax.scatter(i,j,k, s=2, marker='o', color='black', alpha=min(target[n,i,j,k], 1.0))
                        ax.scatter(i,j,k, s=2, marker='o', color='black', alpha=min(features[n,i,j,k], 1.0))

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

        #print("features", features.dtype)
        #print("target", target.dtype)
        #print("aa_coords_intra", aa_coords_intra.dtype)
        #print("aa_coords", aa_coords.dtype)

        elems = (features, target, aa_coords_intra, aa_coords)

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

        if int(cfg.getint('universe', 'n_inter_atoms')) != 0:
            self.n_input = self.ff_aa.n_atom_chns * 2
        else:
            self.n_input = self.ff_aa.n_atom_chns

        self.n_out = self.ff_cg.n_atoms
        self.z_dim = int(cfg.getint('model', 'noise_dim'))

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
        self.prior_schedule = np.array([0] + [int(v) for v in prior_schedule.split(",")])

        self.ratio_bonded_nonbonded = cfg.getfloat('prior', 'ratio_bonded_nonbonded')
        self.prior_mode = cfg.get('prior', 'mode')
        print(self.prior_mode)

        #Model selection
        #if cfg.get('model', 'model_type') == "tiny":
        #    print("Using tiny model")
        if self.z_dim != 0:
            self.generator = model.G_tiny_with_noise(z_dim=self.z_dim,
                                                n_input=self.n_input,
                                                n_output=self.n_out,
                                                start_channels=self.cfg.getint('model', 'n_chns'),
                                                fac=1,
                                                sn=self.cfg.getint('model', 'sn_gen'),
                                                device=device)
            print("Using tiny generator with noise")
        else:
            self.generator = model.G_tiny(n_input=self.n_input,
                                            n_output=self.n_out,
                                            start_channels=self.cfg.getint('model', 'n_chns'),
                                            fac=1,
                                            sn=self.cfg.getint('model', 'sn_gen'),
                                            device=device)
            print("Using tiny generator without noise")

        if cfg.getint('grid', 'resolution') == 8:
            self.critic = model.C_tiny(in_channels=self.n_out,
                                              start_channels=self.cfg.getint('model', 'n_chns'),
                                              fac=1, sn=self.cfg.getint('model', 'sn_crit'),
                                              device=device)
            print("Using tiny critic with resolution 8")


        else:
            self.critic = model.C_tiny16(in_channels=self.n_out,
                                              start_channels=self.cfg.getint('model', 'n_chns'),
                                              fac=1, sn=self.cfg.getint('model', 'sn_crit'),
                                              device=device)

            print("Using tiny critic with resolution 16")


        self.use_gp = cfg.getboolean('model', 'gp')
        self.use_ol = cfg.getboolean('model', 'ol')


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

    def prior_weight(self):
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
        return overlap_loss

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

    def get_energies_cg(self, atom_grid, energy_ndx):
        coords = avg_blob(
            atom_grid,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma_cg'),
            device=self.device,
        )
        #print(coords)
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
        if lj_ndx.size()[1]:
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
        #loss_epoch = [[]]*11

        batch = next(self.loader_train)
        train_batch = self.map_to_device(train_batch)
        elems, energy_ndx_aa, energy_ndx_cg = train_batch
        features, target, _, _ = elems

        target = target[0:1]
        print(target.size())

        tqdm_train_iterator = tqdm(self.loader_train, total=steps_per_epoch, leave=False)
        for train_batch in tqdm_train_iterator:

            train_batch = self.map_to_device(train_batch)
            elems, energy_ndx_aa, energy_ndx_cg = train_batch

            features, _, _, _ = elems
            print(features.size())
            print(target.size())
            ol = self.overlap_loss(features, target)
            print(ol)

    def val(self):
        resolution = self.cfg.getint('grid', 'resolution')
        grid_length = self.cfg.getfloat('grid', 'length')
        delta_s = self.cfg.getfloat('grid', 'length') / self.cfg.getint('grid', 'resolution')
        sigma_aa = self.cfg.getfloat('grid', 'sigma_aa')
        sigma_cg = self.cfg.getfloat('grid', 'sigma_cg')
        grid = torch.from_numpy(make_grid_np(delta_s, resolution)).to(self.device)

        g = Mol_Generator(self.data, train=True, rand_rot=False)
        all_elems = list(g)

        print("jetzt gehts los")
        n = 0
        try:
            self.generator.eval()
            self.critic.eval()
            print("oha")
            for g in all_elems:
                mol = g['aa_mol']

                aa_positions_intra = np.array([g['aa_positions_intra']])
                aa_intra_featvec = np.array([g['aa_intra_featvec']])

                aa_positions_intra = torch.from_numpy(aa_positions_intra).to(self.device).float()
                aa_blobbs_intra = self.to_voxel(aa_positions_intra, grid, sigma_aa)

                # print(aa_intra_featvec[:, :, :, None, None, None].shape)
                # print(aa_blobbs_intra[:, :, None, :, :, :].size())
                features = torch.from_numpy(aa_intra_featvec[:, :, :, None, None, None]).to(
                    self.device) * aa_blobbs_intra[:, :, None, :, :, :]
                features = torch.sum(features, 1)

                ol_min_glob = 100.0
                print(n)
                for ndx in range(0, len(all_elems), self.bs):
                    with torch.no_grad():
                        batch = all_elems[ndx:min(ndx + self.bs, len(all_elems))]

                        #print(batch)

                        cg_positions_intra = np.array([d['cg_positions_intra'] for d in batch])
                        cg_positions_intra = torch.from_numpy(cg_positions_intra).to(self.device).float()
                        target = self.to_voxel(cg_positions_intra, grid, sigma_cg)

                        ol = self.overlap_loss(features, target)
                        ol = ol.detach().cpu().numpy()
                        ndx = ol.argmin()

                        ol_min = ol[ndx]
                        if ol_min < ol_min_glob:
                            ol_min_glob = ol_min
                            min_coords = np.array([d['cg_positions_intra'] for d in batch])[ndx]

                        #print(ol)
                        """
                        coords = avg_blob(
                            fake_mol,
                            res=resolution,
                            width=grid_length,
                            sigma=sigma_cg,
                            device=self.device,)
                        """

                min_coords = np.dot(min_coords, mol.rot_mat.T)
                for pos, bead in zip(min_coords, mol.beads):
                    bead.pos = pos + mol.com
                n = n +1
            samples_dir = self.out.output_dir / "samples"
            samples_dir.mkdir(exist_ok=True)

            for sample in self.data.samples_train_aa:
                #sample.write_gro_file(samples_dir / (sample.name + str(self.step) + ".gro"))
                sample.write_gro_file(samples_dir / (sample.name + ".gro"))
        finally:
            self.generator.train()
            self.critic.train()



    def to_voxel(self, coords, grid, sigma):
        coords = coords[..., None, None, None]
        return torch.exp(-1.0 * torch.sum((grid - coords) * (grid - coords), axis=2) / sigma).float()


