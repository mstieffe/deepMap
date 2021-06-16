import numpy as np
import math
import matplotlib.pyplot as plt
import random

class Mol_N_Generator_AA():

    def __init__(self, data, train=False, rand_rot=False, n_mols=3):

        self.data = data
        self.train = train
        self.rand_rot = rand_rot
        self.n_mols = n_mols

        if train:
            self.samples_inp = self.data.samples_train_inp
            #self.samples_out = self.data.samples_train_out
        else:
            self.samples_inp = self.data.samples_val_inp
            #self.samples_out = self.data.samples_val_out
        self.mols_inp, self.mols_out = [], []
        for s in self.samples_inp:
            self.mols_inp += s.mols
        #for s in self.samples_out:
        #    self.mols_out += s.mols
        #for mols in [s.mols for s in self.samples_inp]:
        #    self.mols_inp += mols
        #for mols in [s.mols for s in self.samples_out]:
        #    self.mols_out += mols

        random.shuffle(self.mols_inp)

    def __iter__(self):

        #go through every mol
        for mol_inp in self.mols_inp:
            d = {}

            batch_mol_inp = mol_inp.env_mols[:self.n_mols]
            env_mol_inp = mol_inp.env_mols[self.n_mols:]

            positions_intra_inp, atoms_inp = [], []
            for mol_inp in batch_mol_inp:
                for a in mol_inp.atoms:
                    atoms_inp.append(a)
                    positions_intra_inp.append(mol_inp.box.diff_vec(a.pos - mol_inp.com))

            positions_inter_inp, positions_inter_out, env_atoms_inp, env_atoms_out = [], [], [], []
            for mol in env_mol_inp:
                for a in mol.atoms:
                    env_atoms_inp.append(a)
                    positions_inter_inp.append(mol_inp.box.diff_vec(a.pos - mol_inp.com))
                for b in mol.beads:
                    positions_inter_out.append(mol_inp.box.diff_vec(b.pos - mol_inp.com))

            #align
            positions_intra_inp = np.dot(positions_intra_inp, mol_inp.rot_mat)
            if self.data.n_env_mols:
                positions_inter_inp = np.dot(positions_inter_inp, mol_inp.rot_mat)

                positions_inter_out = np.dot(positions_inter_out, mol_inp.rot_mat)

            """
            atoms = list(mol_inp.atoms) + list(mol_inp.intermolecular_atoms)

            #energy ndx
            
            inp_intra_index_dict = dict(zip(mol_inp.atoms, range(0, len(mol_inp.atoms))))
            inp_index_dict = dict(zip(atoms, range(0, len(atoms))))

            inp_bond_ndx, inp_ang_ndx, inp_dih_ndx, inp_lj_intra_ndx,  inp_lj_ndx = self.energy_ndx(mol_inp, inp_intra_index_dict, inp_index_dict)
            """

            #features (atom types)
            inp_intra_featvec = self.featvec(atoms_inp, self.data.ff_inp)
            inp_inter_featvec = self.featvec(env_atoms_inp, self.data.ff_inp)

            out_inter_featvec = self.featvec(env_atoms_out, self.data.ff_out)


            d={
                "inp_positions_intra": np.array(positions_intra_inp, dtype=np.float32),
                "inp_positions_inter": np.array(positions_inter_inp, dtype=np.float32),
                "out_positions_inter": np.array(positions_inter_out, dtype=np.float32),
                "inp_intra_featvec": np.array(inp_intra_featvec, dtype=np.float32),
                "inp_inter_featvec": np.array(inp_inter_featvec, dtype=np.float32),
                "out_inter_featvec": np.array(out_inter_featvec, dtype=np.float32),
                "inp_mols": batch_mol_inp
            }

            """
            fig = plt.figure(figsize=(20,20))
            n_chns = 4
            colours = ['red', 'black', 'green', 'blue']

            coords = [np.array(positions_intra_inp), np.array(positions_inter_inp), np.array(positions_intra_out), np.array(positions_inter_out)]
            features = [list(inp_intra_featvec), list(inp_inter_featvec), list(out_intra_featvec), list(out_inter_featvec)]
            for c in range(0, n_chns):

                ax = fig.add_subplot(int(np.ceil(np.sqrt(n_chns))),int(np.ceil(np.sqrt(n_chns))), c+1, projection='3d')
                pos = coords[c]
                #ax.scatter(mol_inp.com[0], mol_inp.com[1],mol_inp.com[2], s=20, marker='o', color='blue', alpha=0.5)
                for n in range(0, len(pos)):
                    colour_ndx = features[c][n].tolist().index(1.0)
                    ax.scatter(pos[n,0], pos[n,1], pos[n,2], s=5, marker='o', color=colours[colour_ndx], alpha = 0.5)
                ax.set_xlim3d(-1.0, 1.0)
                ax.set_ylim3d(-1.0, 1.0)
                ax.set_zlim3d(-1.0, 1.0)
                ax.set_xticks(np.arange(-1, 1, step=0.5))
                ax.set_yticks(np.arange(-1, 1, step=0.5))
                ax.set_zticks(np.arange(-1, 1, step=0.5))
                ax.tick_params(labelsize=6)
                plt.plot([0.0, 0.0], [0.0, 0.0], [-1.0, 1.0])
            plt.show()
            """

            yield d

    def featvec(self, atoms, ff):
        featvec = np.zeros((len(atoms), ff.n_atom_chns))
        for index in range(0, len(atoms)):
            #print(atoms[index].type.channel, ff.atom_types[atoms[index].type.name].channel)
            featvec[index, ff.atom_types[atoms[index].type.name].channel] = 1
        return featvec

    def energy_ndx(self, mol, index_intra_dict, index_dict):
        bond_ndx = []
        for b in mol.bonds:
            bond_ndx.append(tuple([b.type_index,
                                   index_intra_dict[b.atoms[0]],
                                   index_intra_dict[b.atoms[1]]]))

        angle_ndx = []
        for a in mol.angles:
            angle_ndx.append(tuple([a.type_index,
                                    index_intra_dict[a.atoms[0]],
                                    index_intra_dict[a.atoms[1]],
                                    index_intra_dict[a.atoms[2]]]))

        dih_ndx = []
        for d in mol.dihs:
            dih_ndx.append(tuple([d.type_index,
                                  index_intra_dict[d.atoms[0]],
                                  index_intra_dict[d.atoms[1]],
                                  index_intra_dict[d.atoms[2]],
                                  index_intra_dict[d.atoms[3]]]))

        lj_intra_ndx = []
        for l in mol.ljs_intra:
            lj_intra_ndx.append(tuple([l.type_index,
                                 index_intra_dict[l.atoms[0]],
                                 index_intra_dict[l.atoms[1]]]))

        lj_ndx = []
        for l in mol.ljs:
            lj_ndx.append(tuple([l.type_index,
                                 index_dict[l.atoms[0]],
                                 index_dict[l.atoms[1]]]))

        return bond_ndx, angle_ndx, dih_ndx, lj_intra_ndx, lj_ndx

    def all_elems(self):
        g = iter(self)
        elems = []
        for e in g:
            elems.append(e)
        return elems


    def rand_rot_mat(self):
        #rotation axis
        if self.data.align:
            v_rot = np.array([0.0, 0.0, 1.0])
        else:
            phi = np.random.uniform(0, np.pi * 2)
            costheta = np.random.uniform(-1, 1)
            theta = np.arccos(costheta)
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            v_rot = np.array([x, y, z])

        #rotation angle
        theta = np.random.uniform(0, np.pi * 2)

        #rotation matrix
        a = math.cos(theta / 2.0)
        b, c, d = -v_rot * math.sin(theta / 2.0)
        inp, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        rot_mat = np.array([[inp + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), inp + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), inp + dd - bb - cc]])

        rot_mat = rot_mat.astype('float32')

        return rot_mat

    def pad1d(self, vec, max, value=0):
        vec = np.pad(vec, (0, max - len(vec)), 'constant', constant_values=(0, value))
        return vec

    def pad2d(self, vec, max, value=0):
        vec = np.pad(vec, ((0, max - len(vec)), (0, 0)), 'constant', constant_values=(0, value))
        return vec

    def pad_energy_ndx(self, ndx, max, value=tuple([-1, 1, 2])):
        #remove dupicates
        ndx = list(set(ndx))
        #pad
        for n in range(0, max - len(ndx)):
            ndx.append(tuple(value))
        return ndx