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
        #self.loader_val = cycle(loader_val)
        #self.val_data = ds_val.data

        #self.n_gibbs = int(cfg.getint('validate', 'n_gibbs'))

        #model
        self.name = cfg.get('model', 'name')
        #self.z_dim = int(cfg.getint('model', 'noise_dim'))
        if int(cfg.getint('universe', 'n_inter_atoms')) != 0:
            self.n_input = self.ff_aa.n_atom_chns * 2
        else:
            self.n_input = self.ff_aa.n_atom_chns

        self.n_out = self.ff_cg.n_atoms
        #self.z_and_label_dim = self.z_dim + self.n_atom_chns

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

        #self.prior_weights = self.get_prior_weights()
        self.ratio_bonded_nonbonded = cfg.getfloat('prior', 'ratio_bonded_nonbonded')
        #self.lj_weight = cfg.getfloat('training', 'lj_weight')
        #self.covalent_weight = cfg.getfloat('training', 'covalent_weight')
        self.prior_mode = cfg.get('prior', 'mode')
        print(self.prior_mode)
        #self.w_prior = torch.tensor(self.prior_weights[self.step], dtype=torch.float32, device=device)

        #Model selection
        #if cfg.get('model', 'model_type') == "tiny":
        #    print("Using tiny model")
        self.generator = model.AtomGen_mid(n_input=self.n_input,
                                            n_output=self.n_out,
                                            start_channels=self.cfg.getint('model', 'n_chns'),
                                            fac=1,
                                            sn=self.cfg.getint('model', 'sn_gen'),
                                            device=device)

        if cfg.getint('grid', 'resolution') == 8:
            self.critic = model.AtomCrit_tiny(in_channels=self.n_out,
                                              start_channels=self.cfg.getint('model', 'n_chns'),
                                              fac=1, sn=self.cfg.getint('model', 'sn_crit'),
                                              device=device)

        else:
            self.critic = model.AtomCrit_tiny16(in_channels=self.n_out,
                                              start_channels=self.cfg.getint('model', 'n_chns'),
                                              fac=1, sn=self.cfg.getint('model', 'sn_crit'),
                                              device=device)



        self.use_gp = cfg.getboolean('model', 'gp')
        self.use_ol = cfg.getboolean('model', 'ol')

        #self.mse = torch.nn.MSELoss()
        #self.kld = torch.nn.KLDivLoss(reduction="batchmean")

        self.critic.to(device=device)
        self.generator.to(device=device)
        #self.mse.to(device=device)

        lr_gen = cfg.getfloat('training', 'lr_gen')
        lr_crit = cfg.getfloat('training', 'lr_crit')
        self.opt_generator_pretrain = Adam(self.generator.parameters(), lr=lr_gen, betas=(0, 0.9))
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

    def get_prior_weights(self):
        steps_per_epoch = len(self.loader_train)
        tot_steps = steps_per_epoch * self.cfg.getint('training', 'n_epoch')

        prior_weights = self.cfg.get('training', 'energy_prior_weights')
        prior_weights = [float(v) for v in prior_weights.split(",")]
        prior_steps = self.cfg.get('training', 'n_start_prior')
        prior_steps = [int(v) for v in prior_steps.split(",")]
        n_trans = self.cfg.getint('training', 'n_prior_transition')
        weights = []
        for s in range(self.step, self.step + tot_steps):
            if s > prior_steps[-1]:
                ndx = len(prior_weights)-1
                #weights.append(self.energy_prior_values[-1])
            else:
                for n in range(0, len(prior_steps)):
                    if s < prior_steps[n]:
                        #weights.append(self.energy_prior_values[n])
                        ndx = n
                        break
            #print(ndx)
            if ndx > 0 and s < prior_steps[ndx-1] + self.cfg.getint('training', 'n_prior_transition'):
                weights.append(prior_weights[ndx-1] + (prior_weights[ndx]-prior_weights[ndx-1])*(s-prior_steps[ndx-1])/n_trans)
            else:
                weights.append(prior_weights[ndx])

        return weights


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

    def transpose_and_zip(self, args):
        args = tuple(torch.transpose(x, 0, 1) for x in args)
        elems = zip(*args)
        return elems

    def featurize(self, grid, features):
        grid = grid[:, :, None, :, :, :] * features[:, :, :, None, None, None]
        #grid (BS, N_atoms, 1, N_x, N_y, N_z) * features (BS, N_atoms, N_features, 1, 1, 1)
        return torch.sum(grid, 1)

    def prepare_condition(self, fake_atom_grid, real_atom_grid, aa_featvec, bead_features):
        fake_aa_features = self.featurize(fake_atom_grid, aa_featvec)
        real_aa_features = self.featurize(real_atom_grid, aa_featvec)
        c_fake = fake_aa_features + bead_features
        c_real = real_aa_features + bead_features
        return c_fake, c_real

    def generator_loss(self, critic_fake):
        return (-1.0 * critic_fake).mean()

    def overlap_loss(self, aa_mol, cg_mol):
        aa_mol = torch.flatten(torch.sum(aa_mol, 1), 1)
        aa_mol = aa_mol / torch.sum(aa_mol, 1, keepdim=True)
        cg_mol = torch.flatten(torch.sum(cg_mol, 1), 1)
        cg_mol = cg_mol / torch.sum(cg_mol, 1, keepdim=True)

        overlap_loss = (aa_mol * (aa_mol / cg_mol).log()).sum(1)
        return torch.mean(overlap_loss)

    def overlap_loss2(self, aa_mol, cg_mol):
        aa_mol = torch.sum(aa_mol, 1)
        aa_mol = aa_mol / torch.sum(aa_mol, (1, 2, 3), keepdim=True)
        cg_mol = torch.sum(cg_mol, 1)
        cg_mol = cg_mol / torch.sum(cg_mol, (1, 2, 3), keepdim=True)
        overlap_loss = aa_mol * cg_mol
        overlap_loss = torch.sum(overlap_loss, (1,2,3))
        overlap_loss = -torch.mean(overlap_loss, 0)
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
        for epoch in epochs:
            n = 0
            #loss_epoch = [[]]*11
            val_iterator = iter(self.loader_val)
            tqdm_train_iterator = tqdm(self.loader_train, total=steps_per_epoch, leave=False)
            for train_batch in tqdm_train_iterator:

                train_batch = self.map_to_device(train_batch)
                elems, energy_ndx_aa, energy_ndx_cg = train_batch

                if n == n_critic:
                    g_loss_dict = self.train_step_gen(elems, energy_ndx_aa, energy_ndx_cg)
                    #print(g_loss_dict)
                    for key, value in g_loss_dict.items():
                        self.out.add_scalar(key, value, global_step=self.step)
                    tqdm_train_iterator.set_description('D: {:.2f}, G: {:.2f}, E_cg: {:.2f}, {:.2f}, {:.2f}, {:.2f}, E_aa: {:.2f}, {:.2f}, {:.2f}, {:.2f}, OL: {:.2f}'.format(c_loss,
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
                    c_loss = self.train_step_critic(elems)
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
        resolution = self.cfg.getint('grid', 'resolution')
        delta_s = self.cfg.getfloat('grid', 'length') / self.cfg.getint('grid', 'resolution')
        sigma = self.cfg.getfloat('grid', 'sigma_cg')
        grid = torch.from_numpy(make_grid_np(delta_s, resolution)).to(self.device)

        g = Mol_Generator_AA(self.data, train=False, rand_rot=False)
        all_elems = list(g)
        for ndx in range(0, len(all_elems), self.bs):
            with torch.no_grad():
                batch = all_elems[ndx:min(ndx + self.bs, len(all_elems))]

                aa_positions_intra = np.array([d['aa_positions_intra'] for d in batch])
                aa_intra_featvec = np.array([d['aa_intra_featvec'] for d in batch])

                mols = np.array([d['aa_mol'] for d in batch])

                aa_positions_intra = torch.from_numpy(aa_positions_intra).to(self.device).float()
                aa_blobbs_intra = self.to_voxel(aa_positions_intra, grid, sigma)

                #print(aa_intra_featvec[:, :, :, None, None, None].shape)
                #print(aa_blobbs_intra[:, :, None, :, :, :].size())
                features = torch.from_numpy(aa_intra_featvec[:, :, :, None, None, None]).to(self.device) * aa_blobbs_intra[:, :, None, :, :, :]
                features = torch.sum(features, 1)

                #elems, energy_ndx_aa, energy_ndx_cg = val_batch
                #features, _, aa_coords_intra, aa_coords = elems
                fake_mol = self.generator(features)

                coords = avg_blob(
                    fake_mol,
                    res=self.cfg.getint('grid', 'resolution'),
                    width=self.cfg.getfloat('grid', 'length'),
                    sigma=self.cfg.getfloat('grid', 'sigma_cg'),
                    device=self.device,)
                for positions, mol in zip(coords, mols):
                    positions = positions.detach().cpu().numpy()
                    positions = np.dot(positions, mol.rot_mat.T)
                    for pos, bead in zip(positions, mol.beads):
                        bead.pos = pos + mol.com

        samples_dir = self.out.output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        for sample in self.data.samples_val_aa:
            #sample.write_gro_file(samples_dir / (sample.name + str(self.step) + ".gro"))
            sample.write_gro_file(samples_dir / (sample.name + ".gro"))

    def train_step_critic(self, elems):

        features, target, _, _ = elems

        #c_loss = torch.zeros([], dtype=torch.float32, device=self.device)

        #prepare input for generator
        """
        z = torch.empty(
            [target_atom.shape[0], self.z_dim],
            dtype=torch.float32,
            device=self.device,
        ).normal_()
        """

        #generate fake atom
        fake_mol = self.generator(features)

        #fake_data = torch.cat([fake_atom, features], dim=1)
        #real_data = torch.cat([target_atom[:, None, :, :, :], features], dim=1)

        #critic
        critic_fake = self.critic(fake_mol)
        critic_real = self.critic(target)

        #loss
        c_wass = self.critic_loss(critic_real, critic_fake)
        c_eps = self.epsilon_penalty(1e-3, critic_real)
        c_loss = c_wass + c_eps
        if self.use_gp:
            c_gp = self.gradient_penalty(target, fake_mol)
            c_loss += c_gp

        self.opt_critic.zero_grad()
        c_loss.backward()
        self.opt_critic.step()

        return c_loss.detach().cpu().numpy()


    def train_step_gen(self, elems, energy_ndx_aa, energy_ndx_cg, backprop=True):

        features, target, aa_coords_intra, aa_coords = elems


        #g_wass = torch.zeros([], dtype=torch.float32, device=self.device)

        """
        z = torch.empty(
            [target_atom.shape[0], self.z_dim],
            dtype=torch.float32,
            device=self.device,
        ).normal_()
        """

        #generate fake atom
        fake_mol = self.generator(features)

        #critic
        #critic_fake = self.critic(torch.cat([fake_atom, features], dim=1))

        #mask
        #critic_fake = torch.squeeze(critic_fake)

        #loss
        g_wass = self.generator_loss(fake_mol)
        g_overlap = self.overlap_loss(features, fake_mol)
        if self.use_ol:
            g_loss = g_wass + self.ol_weight * g_overlap
        else:
            g_loss = g_wass

        #g_loss = g_overlap

        #real_atom_grid = torch.where(repl[:, :, None, None, None], atom_grid, target_atom[:, None, :, :, :])
        #fake_atom_grid = torch.where(repl[:, :, None, None, None], atom_grid, fake_atom)

        e_bond_cg, e_angle_cg, e_dih_cg, e_lj_cg = self.get_energies_cg(fake_mol, energy_ndx_cg)
        e_bond_aa, e_angle_aa, e_dih_aa, e_lj_aa = self.get_energies_aa(aa_coords_intra, aa_coords, energy_ndx_aa)

        #if 1:
        #    g_loss += e_bond_cg + e_angle_cg + e_dih_cg + e_lj_cg

        #g_loss = g_wass + self.prior_weight() * energy_loss
        #g_loss = g_wass

        if backprop:
            self.opt_generator.zero_grad()
            g_loss.backward()
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

    def repeat(self, t):
        return tuple(torch.stack(self.bs*[x]) for x in t)

    def to_voxel(self, coords, grid, sigma):
        coords = coords[..., None, None, None]
        return torch.exp(-1.0 * torch.sum((grid - coords) * (grid - coords), axis=2) / sigma).float()

    def predict(self, elems, initial, energy_ndx):

        aa_grid, cg_features = initial

        generated_atoms = []
        for target_type, aa_featvec, repl in zip(*elems):
            fake_aa_features = self.featurize(aa_grid, aa_featvec)
            c_fake = fake_aa_features + cg_features
            target_type = target_type.repeat(self.bs, 1)
            z = torch.empty(
                [target_type.shape[0], self.z_dim],
                dtype=torch.float32,
                device=self.device,
            ).normal_()

            #generate fake atom
            fake_atom = self.generator(z, target_type, c_fake)
            generated_atoms.append(fake_atom)

            #update aa grids
            aa_grid = torch.where(repl[:,:,None,None,None], aa_grid, fake_atom)

        #generated_atoms = torch.stack(generated_atoms, dim=1)
        generated_atoms = torch.cat(generated_atoms, dim=1)

        generated_atoms_coords = avg_blob(
            generated_atoms,
            res=self.cfg.getint('grid', 'resolution'),
            width=self.cfg.getfloat('grid', 'length'),
            sigma=self.cfg.getfloat('grid', 'sigma'),
            device=self.device,
        )

        b_energy, a_energy, d_energy, l_energy = self.get_energies_from_grid(aa_grid, energy_ndx)
        energy = b_energy + a_energy + d_energy + l_energy

        return generated_atoms_coords, energy

    def validate(self, samples_dir=None):

        if samples_dir:
            samples_dir = self.out.output_dir / samples_dir
            make_dir(samples_dir)
        else:
            samples_dir = self.out.samples_dir
        stats = Stats(self.data, dir= samples_dir / "stats")

        print("Saving samples in {}".format(samples_dir), "...", end='')

        resolution = self.cfg.getint('grid', 'resolution')
        delta_s = self.cfg.getfloat('grid', 'length') / self.cfg.getint('grid', 'resolution')
        sigma = self.cfg.getfloat('grid', 'sigma')
        #grid = make_grid_np(delta_s, resolution)

        grid = torch.from_numpy(make_grid_np(delta_s, resolution)).to(self.device)
        rot_mtxs = torch.from_numpy(rot_mtx_batch(self.bs)).to(self.device).float()
        rot_mtxs_transposed = torch.from_numpy(rot_mtx_batch(self.bs, transpose=True)).to(self.device).float()

        data_generators = []
        data_generators.append(iter(Recurrent_Generator(self.data, hydrogens=False, gibbs=False, train=False, rand_rot=False, pad_seq=False, ref_pos=False)))
        data_generators.append(iter(Recurrent_Generator(self.data, hydrogens=True, gibbs=False, train=False, rand_rot=False, pad_seq=False, ref_pos=False)))

        for m in range(self.n_gibbs):
            data_generators.append(iter(Recurrent_Generator(self.data, hydrogens=False, gibbs=True, train=False, rand_rot=False, pad_seq=False, ref_pos=False)))
            data_generators.append(iter(Recurrent_Generator(self.data, hydrogens=True, gibbs=True, train=False, rand_rot=False, pad_seq=False, ref_pos=False)))

        try:
            self.generator.eval()
            self.critic.eval()

            for data_gen in data_generators:
                start = timer()

                for d in data_gen:
                    with torch.no_grad():

                        aa_coords = torch.matmul(torch.from_numpy(d['aa_pos']).to(self.device).float(), rot_mtxs)
                        cg_coords = torch.matmul(torch.from_numpy(d['cg_pos']).to(self.device).float(), rot_mtxs)

                        #aa_coords = torch.from_numpy(d['aa_pos']).to(self.device).float()
                        #cg_coords = torch.from_numpy(d['cg_pos']).to(self.device).float()

                        aa_grid = self.to_voxel(aa_coords, grid, sigma)
                        cg_grid = self.to_voxel(cg_coords, grid, sigma)

                        cg_features = torch.from_numpy(d['cg_feat'][None, :, :, None, None, None]).to(self.device) * cg_grid[:, :, None, :, :, :]
                        cg_features = torch.sum(cg_features, 1)

                        initial = (aa_grid, cg_features)

                        elems = (d['target_type'], d['aa_feat'], d['repl'])
                        elems = self.transpose(self.insert_dim(self.to_tensor(elems)))

                        energy_ndx = (d['bonds_ndx'], d['angles_ndx'], d['dihs_ndx'], d['ljs_ndx'])
                        energy_ndx = self.repeat(self.to_tensor(energy_ndx))

                        new_coords, energies = self.predict(elems, initial, energy_ndx)

                        ndx = energies.argmin()

                        new_coords = torch.matmul(new_coords[ndx], rot_mtxs_transposed[ndx])
                        #new_coords = new_coords[ndx]
                        new_coords = new_coords.detach().cpu().numpy()

                        for c, a in zip(new_coords, d['atom_seq']):

                            a.pos = d['loc_env'].rot_back(c)
                            #a.ref_pos = d['loc_env'].rot_back(c)

                print(timer()-start)
            stats.evaluate(train=False, subdir=str(self.epoch), save_samples=True)
            #reset atom positions
            for sample in self.data.samples_val:
                #sample.write_gro_file(samples_dir / (sample.name + str(self.step) + ".gro"))
                sample.kick_atoms()

        finally:
            self.generator.train()
            self.critic.train()



