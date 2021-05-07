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
        self.n_interatoms = int(cfg.getint('universe', 'n_inter_atoms'))

        #forcefield
        self.ff_aa_name = cfg.get('data', 'ff_aa')
        self.ff_aa_path = Path("./data/ff") / self.ff_aa_name
        self.ff_aa = FF(self.ff_aa_path)

        self.ff_cg_name = cfg.get('data', 'ff_cg')
        self.ff_cg_path = Path("./data/ff") / self.ff_cg_name
        self.ff_cg = FF(self.ff_cg_path)

        self.top_aa = Path("./data/top") / cfg.get('data', 'top_aa')
        self.top_cg = Path("./data/top") / cfg.get('data', 'top_cg')

        self.desc = '_align={}_cutoff={}_kick={}_ff_aa={}_ff_cg={}.pkl'.format(self.align,
                                                                                self.cutoff,
                                                                                self.kick,
                                                                                self.ff_aa_name,
                                                                                self.ff_cg_name)

        #samples
        self.data_dir = Path("./data/")
        self.dirs_train_aa = [Path("./data/reference_snapshots/") / d.replace(" ", "") for d in cfg.get('data', 'train_data_aa').split(",")]
        self.dirs_val_aa = [Path("./data/reference_snapshots/") / d.replace(" ", "") for d in cfg.get('data', 'val_data_aa').split(",")]
        self.dirs_train_cg = [Path("./data/reference_snapshots/") / d.replace(" ", "") for d in cfg.get('data', 'train_data_cg').split(",")]
        self.dirs_val_cg = [Path("./data/reference_snapshots/") / d.replace(" ", "") for d in cfg.get('data', 'val_data_cg').split(",")]
        self.dir_processed = Path("./data/processed")
        self.dir_processed.mkdir(exist_ok=True)

        #self.samples_train, self.samples_val = [], []
        self.dict_train_aa, self.dict_val_aa, self.dict_train_cg, self.dict_val_cg = {}, {}, {}, {}
        for path in self.dirs_train_aa:
            self.dict_train_aa[path.stem] = self.get_samples(path, res="aa", save=save)
        for path in self.dirs_train_cg:
            self.dict_train_cg[path.stem] = self.get_samples(path, res="cg", save=save)
        self.samples_train_aa = list(itertools.chain.from_iterable(self.dict_train_aa.values()))
        self.samples_train_cg = list(itertools.chain.from_iterable(self.dict_train_cg.values()))
        for path in self.dirs_val_aa:
            self.dict_val_aa[path.stem] = self.get_samples(path, res="aa", save=save)
        for path in self.dirs_val_cg:
            self.dict_val_cg[path.stem] = self.get_samples(path, res="cg", save=save)
        self.samples_val_aa = list(itertools.chain.from_iterable(self.dict_val_aa.values()))
        self.samples_val_cg = list(itertools.chain.from_iterable(self.dict_val_cg.values()))

        #find maximums for padding
        #self.max = self.get_max_dict()

        print("Successfully created universe! This took ", timer()-start, "secs")

    def get_samples(self, path, res="aa", save=False):
        name = path.stem + "_" + res + "_" + self.desc
        processed_path = self.dir_processed / name

        if res == "aa":
            ff = self.ff_aa
            path_dict = {'data_dir': self.data_dir, 'top_aa': self.top_aa, 'top_cg': self.top_cg}
        else:
            ff = self.ff_cg
            path_dict = {'data_dir': self.data_dir, 'top_aa': self.top_cg, 'top_cg': self.top_aa}

        if processed_path.exists():
            with open(processed_path, 'rb') as input:
                samples = pickle.load(input)
            print("Loaded train universe from " + str(processed_path))
        else:
            samples = []
            #dir = path / res
            for p in path.glob('*.gro'):
                #path_dict = {'data_dir': self.data_dir, 'path': p, 'file_name': p.stem, 'top_aa': self.top_aa, 'top_cg': self.top_cg}
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
                max_dict['seq_len'] = max(len(sample.aa_seq_heavy[bead]), len(sample.aa_seq_hydrogens[bead]), max_dict['seq_len'])
                max_dict['beads_loc_env'] = max(len(sample.loc_envs[bead].beads), max_dict['beads_loc_env'])
                max_dict['atoms_loc_env'] = max(len(sample.loc_envs[bead].atoms), max_dict['atoms_loc_env'])

                for aa_seq in [sample.aa_seq_heavy[bead], sample.aa_seq_hydrogens[bead]]:
                    bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx = [], [], [], []
                    for atom in aa_seq:
                        f = sample.aa_features[atom]
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
