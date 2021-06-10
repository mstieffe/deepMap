import numpy as np
import os
import pickle
import mdtraj as md
import networkx as nx
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from dbm.ff import *
from dbm.universe import *
from copy import deepcopy
from operator import add
from pathlib import Path


class Data():

    def __init__(self, cfg, save=True):

        start = timer()

        self.cfg = cfg
        self.align = int(cfg.getboolean('universe', 'align'))
        self.cutoff = cfg.getfloat('universe', 'cutoff')
        self.kick = cfg.getfloat('universe', 'kick')
        self.n_env_mols = int(cfg.getint('universe', 'n_env_mols'))
        self.pairs = cfg.getboolean('data', 'pairs')


        #forcefield
        self.ff_inp_name = cfg.get('data', 'ff_inp')
        self.ff_inp_path = Path("./data/ff") / self.ff_inp_name
        self.ff_inp = FF(self.ff_inp_path)

        self.ff_out_name = cfg.get('data', 'ff_out')
        self.ff_out_path = Path("./data/ff") / self.ff_out_name
        self.ff_out = FF(self.ff_out_path)

        self.top_inp = Path("./data/top") / cfg.get('data', 'top_inp')
        self.top_out = Path("./data/top") / cfg.get('data', 'top_out')

        self.desc = '_align={}_cutoff={}_kick={}_ff_inp={}_ff_out={}.pkl'.format(self.align,
                                                                                self.cutoff,
                                                                                self.kick,
                                                                                self.ff_inp_name,
                                                                                self.ff_out_name)

        #samples
        self.data_dir = Path("./data/")
        self.dirs_train_inp = [Path("./data/reference_snapshots/") / d.replace(" ", "") for d in cfg.get('data', 'train_data_inp').split(",")]
        self.dirs_val_inp = [Path("./data/reference_snapshots/") / d.replace(" ", "") for d in cfg.get('data', 'val_data_inp').split(",")]
        self.dirs_train_out = [Path("./data/reference_snapshots/") / d.replace(" ", "") for d in cfg.get('data', 'train_data_out').split(",")]
        self.dirs_val_out = [Path("./data/reference_snapshots/") / d.replace(" ", "") for d in cfg.get('data', 'val_data_out').split(",")]
        self.dir_processed = Path("./data/processed")
        self.dir_processed.mkdir(exist_ok=True)
        self.dir_mapping = Path("./data/mapping/")

        #self.samples_train, self.samples_val = [], []
        self.dict_train_inp, self.dict_val_inp, self.dict_train_out, self.dict_val_out = {}, {}, {}, {}
        if self.pairs:
            for path_inp, path_out in zip(self.dirs_train_inp, self.dirs_train_out):
                self.dict_train_inp[path_inp.stem], self.dict_train_out[path_out.stem] = self.get_sample_pairs(path_inp, path_out, save=save)
        else:
            for path in self.dirs_train_inp:
                self.dict_train_inp[path.stem] = self.get_samples(path, res="inp", save=save)
            for path in self.dirs_train_out:
                self.dict_train_out[path.stem] = self.get_samples(path, res="out", save=save)
        self.samples_train_inp = list(itertools.chain.from_iterable(self.dict_train_inp.values()))
        self.samples_train_out = list(itertools.chain.from_iterable(self.dict_train_out.values()))
        if cfg.getboolean('data', 'pairs'):
            for path_inp, path_out in zip(self.dirs_val_inp, self.dirs_val_out):
                self.dict_val_inp[path_inp.stem], self.dict_val_out[path_out.stem] = self.get_sample_pairs(path_inp, path_out, save=save)
        else:
            for path in self.dirs_val_inp:
                print(path)
                self.dict_val_inp[path.stem] = self.get_samples(path, res="inp", save=save)
                print(self.dict_val_inp[path.stem])
            for path in self.dirs_val_out:
                self.dict_val_out[path.stem] = self.get_samples(path, res="out", save=save)
        self.samples_val_inp = list(itertools.chain.from_iterable(self.dict_val_inp.values()))
        self.samples_val_out = list(itertools.chain.from_iterable(self.dict_val_out.values()))

        #find maximums for padding
        #self.max = self.get_max_dict()

        print("Successfully created universe! This took ", timer()-start, "secs")

    def get_sample_pairs(self, path_inp, path_out, save=False):
        name_inp = path_inp.stem + "_" + self.desc
        processed_path_inp = self.dir_processed / name_inp
        name_out = path_out.stem + "_" + self.desc
        processed_path_out = self.dir_processed / name_out

        for p in path_out.glob('*.gro'):
            inp_file = path_inp / p.name
            if not inp_file.exists():
                raise Exception('can not generate sample pairs. no matching files found for ', inp_file)

        if processed_path_out.exists() and processed_path_inp.exists():
            with open(processed_path_inp, 'rb') as input:
                samples_inp = pickle.load(input)
            print("Loaded train universe from " + str(processed_path_inp))
            with open(processed_path_out, 'rb') as input:
                samples_out = pickle.load(input)
            print("Loaded train universe from " + str(processed_path_out))
        else:
            samples_inp, samples_out = [], []
            # dir = path / res
            for p_inp, p_out in zip(path_inp.glob('*.gro'), path_out.glob('*.gro')):
                # path_dict = {'data_dir': self.data_dir, 'path': p, 'file_name': p.stem, 'top_inp': self.top_inp, 'top_out': self.top_out}
                path_dict_inp = {'data_dir': self.data_dir, 'top_inp': self.top_inp, 'top_out': self.top_out, 'path': p_inp, 'file_name': p_inp.stem}
                path_dict_out = {'data_dir': self.data_dir, 'top_inp': self.top_out, 'top_out': self.top_inp, 'path': p_out, 'file_name': p_out.stem}

                u_inp = Universe(self.cfg, path_dict_inp, self.ff_inp)
                coms = [m.com for m in u_inp.mols]
                u_out = Universe(self.cfg, path_dict_out, self.ff_out, coms=coms)

                if len(u_inp.mols) != len(u_out.mols):
                    raise Exception('number of molecules does not match for ', p_inp, ' and ', p_out)

                samples_inp.append(u_inp)
                samples_out.append(u_out)
            if save:
                with open(processed_path_inp, 'wb') as output:
                    pickle.dump(samples_inp, output, pickle.HIGHEST_PROTOCOL)
                with open(processed_path_out, 'wb') as output:
                    pickle.dump(samples_out, output, pickle.HIGHEST_PROTOCOL)
        return samples_inp, samples_out

    def get_samples(self, path, res="inp", save=False):
        name = path.stem + "_" + res + "_" + self.desc
        processed_path = self.dir_processed / name

        if res == "inp":
            ff = self.ff_inp
            path_dict = {'data_dir': self.data_dir, 'top_inp': self.top_inp, 'top_out': self.top_out}
        else:
            ff = self.ff_out
            path_dict = {'data_dir': self.data_dir, 'top_inp': self.top_out, 'top_out': self.top_inp}

        if processed_path.exists():
            with open(processed_path, 'rb') as input:
                samples = pickle.load(input)
            print("Loaded train universe from " + str(processed_path))
        else:
            samples = []
            #dir = path / res
            for p in path.glob('*.gro'):
                #path_dict = {'data_dir': self.data_dir, 'path': p, 'file_name': p.stem, 'top_inp': self.top_inp, 'top_out': self.top_out}
                path_dict['path'] = p
                path_dict['file_name'] = p.stem
                u = Universe(self.cfg, path_dict, ff)
                samples.append(u)
            if save:
                with open(processed_path, 'wb') as output:
                    pickle.dump(samples, output, pickle.HIGHEST_PROTOCOL)
        return samples

    def get_max_dict(self):
        keys = ['seq_len',
                'beads_loc_env',
                'atoms_loc_env',
                'bonds_per_atom',
                'angles_per_atom',
                'dihs_per_atom',
                'ljs_per_atom',
                'bonds_per_bead',
                'angles_per_bead',
                'dihs_per_bead',
                'ljs_per_bead']
        max_dict = dict([(key, 0) for key in keys])

        samples = self.samples_train + self.samples_val

        for sample in samples:
            for bead in sample.beads:
                max_dict['seq_len'] = max(len(sample.inp_seq_heavy[bead]), len(sample.inp_seq_hydrogens[bead]), max_dict['seq_len'])
                max_dict['beads_loc_env'] = max(len(sample.loc_envs[bead].beads), max_dict['beads_loc_env'])
                max_dict['atoms_loc_env'] = max(len(sample.loc_envs[bead].atoms), max_dict['atoms_loc_env'])

                for inp_seq in [sample.inp_seq_heavy[bead], sample.inp_seq_hydrogens[bead]]:
                    bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx = [], [], [], []
                    for atom in inp_seq:
                        f = sample.inp_features[atom]
                        max_dict['bonds_per_atom'] = max(len(f.energy_ndx_gibbs['bonds']), max_dict['bonds_per_atom'])
                        max_dict['angles_per_atom'] = max(len(f.energy_ndx_gibbs['angles']), max_dict['angles_per_atom'])
                        max_dict['dihs_per_atom'] = max(len(f.energy_ndx_gibbs['dihs']), max_dict['dihs_per_atom'])
                        max_dict['ljs_per_atom'] = max(len(f.energy_ndx_gibbs['ljs']), max_dict['ljs_per_atom'])
                        bonds_ndx += f.energy_ndx_gibbs['bonds']
                        angles_ndx += f.energy_ndx_gibbs['angles']
                        dihs_ndx += f.energy_ndx_gibbs['dihs']
                        ljs_ndx += f.energy_ndx_gibbs['ljs']
                    max_dict['bonds_per_bead'] = max(len(set(bonds_ndx)), max_dict['bonds_per_bead'])
                    max_dict['angles_per_bead'] = max(len(set(angles_ndx)), max_dict['angles_per_bead'])
                    max_dict['dihs_per_bead'] = max(len(set(dihs_ndx)), max_dict['dihs_per_bead'])
                    max_dict['ljs_per_bead'] = max(len(set(ljs_ndx)), max_dict['ljs_per_bead'])

        return max_dict
