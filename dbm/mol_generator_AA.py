import numpy as np
import math
import matplotlib.pyplot as plt
import random

class Mol_Generator_AA():

    def __init__(self, data, train=False, rand_rot=False):

        self.data = data
        self.train = train
        self.rand_rot = rand_rot

        if train:
            self.samples_aa = self.data.samples_train_aa
            #self.samples_cg = self.data.samples_train_cg
        else:
            self.samples_aa = self.data.samples_val_aa
            #self.samples_cg = self.data.samples_val_cg
        self.mols_aa, self.mols_cg = [], []
        for s in self.samples_aa:
            self.mols_aa += s.mols
        #for s in self.samples_cg:
        #    self.mols_cg += s.mols
        #for mols in [s.mols for s in self.samples_aa]:
        #    self.mols_aa += mols
        #for mols in [s.mols for s in self.samples_cg]:
        #    self.mols_cg += mols

        random.shuffle(self.mols_aa)

    def __iter__(self):

        #go through every mol
        for mol_aa in self.mols_aa:
            d = {}

            positions_intra_aa = []
            for a in mol_aa.atoms:
                positions_intra_aa.append(mol_aa.box.diff_vec(a.pos - mol_aa.com))

            positions_inter_aa, positions_inter_cg = [], []
            for a in mol_aa.intermolecular_atoms:
                positions_inter_aa.append(mol_aa.box.diff_vec(a.pos - mol_aa.com))

            for a in mol_aa.intermolecular_beads:
                positions_inter_cg.append(mol_aa.box.diff_vec(a.pos - mol_aa.com))

            #align
            positions_intra_aa = np.dot(positions_intra_aa, mol_aa.rot_mat)
            if self.data.n_env_mols:
                positions_inter_aa = np.dot(positions_inter_aa, mol_aa.rot_mat)

                positions_inter_cg = np.dot(positions_inter_cg, mol_aa.rot_mat)


            atoms = list(mol_aa.atoms) + list(mol_aa.intermolecular_atoms)

            #energy ndx
            aa_intra_index_dict = dict(zip(mol_aa.atoms, range(0, len(mol_aa.atoms))))
            aa_index_dict = dict(zip(atoms, range(0, len(atoms))))

            aa_bond_ndx, aa_ang_ndx, aa_dih_ndx, aa_lj_intra_ndx,  aa_lj_ndx = self.energy_ndx(mol_aa, aa_intra_index_dict, aa_index_dict)

            #features (atom types)
            aa_intra_featvec = self.featvec(mol_aa.atoms, self.data.ff_aa)
            aa_inter_featvec = self.featvec(mol_aa.intermolecular_atoms, self.data.ff_aa)

            cg_inter_featvec = self.featvec(mol_aa.intermolecular_beads, self.data.ff_cg)


            d={
                "aa_positions_intra": np.array(positions_intra_aa, dtype=np.float32),
                "aa_positions_inter": np.array(positions_inter_aa, dtype=np.float32),
                "cg_positions_inter": np.array(positions_inter_cg, dtype=np.float32),
                "aa_intra_featvec": np.array(aa_intra_featvec, dtype=np.float32),
                "aa_inter_featvec": np.array(aa_inter_featvec, dtype=np.float32),
                "cg_inter_featvec": np.array(cg_inter_featvec, dtype=np.float32),
                "aa_bond_ndx": np.array(aa_bond_ndx, dtype=np.int64),
                "aa_ang_ndx": np.array(aa_ang_ndx, dtype=np.int64),
                "aa_dih_ndx": np.array(aa_dih_ndx, dtype=np.int64),
                "aa_lj_ndx": np.array(aa_lj_ndx, dtype=np.int64),
                "aa_lj_intra_ndx": np.array(aa_lj_intra_ndx, dtype=np.int64),
                "aa_mol": mol_aa
            }

            """
            fig = plt.figure(figsize=(20,20))
            n_chns = 4
            colours = ['red', 'black', 'green', 'blue']

            coords = [np.array(positions_intra_aa), np.array(positions_inter_aa), np.array(positions_intra_cg), np.array(positions_inter_cg)]
            features = [list(aa_intra_featvec), list(aa_inter_featvec), list(cg_intra_featvec), list(cg_inter_featvec)]
            for c in range(0, n_chns):

                ax = fig.add_subplot(int(np.ceil(np.sqrt(n_chns))),int(np.ceil(np.sqrt(n_chns))), c+1, projection='3d')
                pos = coords[c]
                #ax.scatter(mol_aa.com[0], mol_aa.com[1],mol_aa.com[2], s=20, marker='o', color='blue', alpha=0.5)
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
            featvec[index, atoms[index].type.channel] = 1
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
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        rot_mat = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

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