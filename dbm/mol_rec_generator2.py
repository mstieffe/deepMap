import numpy as np
import math
import matplotlib.pyplot as plt
import random
import networkx as nx

class Mol_Rec_Generator():

    def __init__(self, data, train=False, res='out', rand_rot=False):

        self.data = data
        self.train = train
        self.rand_rot = rand_rot

        if res == 'out':
            self.ff = self.data.ff_out
            if train:
                self.samples = self.data.samples_train_out
            else:
                self.samples = self.data.samples_val_out
        else:
            self.ff = self.data.ff_inp
            if train:
                self.samples = self.data.samples_train_inp
            else:
                self.samples = self.data.samples_val_inp

        self.mols = []
        for s in self.samples:
            self.mols += s.mols

        #for mols in [s.mols for s in self.samples_inp]:
        #    self.mols_inp += mols
        #for mols in [s.mols for s in self.samples_out]:
        #    self.mols_out += mols

        #if self.train:
        random.shuffle(self.mols)


    def __iter__(self):

        #go through every mol
        for mol in self.mols:
            d = {}

            positions_intra = []
            for a in mol.atoms:
                positions_intra.append(mol.box.diff_vec(a.pos - mol.com))

            positions_inter = []
            for a in mol.intermolecular_atoms:
                positions_inter.append(mol.box.diff_vec(a.pos - mol.com))


            #align
            positions_intra = np.dot(positions_intra, mol.rot_mat)
            if self.data.n_env_mols:
                positions_inter = np.dot(positions_inter, mol.rot_mat)

            positions = np.concatenate((positions_intra, positions_inter))

            atoms = list(mol.atoms) + list(mol.intermolecular_atoms)

            #energy ndx
            intra_index_dict = dict(zip(mol.atoms, range(0, len(mol.atoms))))
            index_dict = dict(zip(atoms, range(0, len(atoms))))


            bond_ndx, ang_ndx, dih_ndx, lj_intra_ndx,  lj_ndx = self.energy_ndx(mol, intra_index_dict, index_dict)

            atom_seq = list(nx.dfs_preorder_nodes(mol.G, mol.atoms[0]))
            featvecs, repls, targets = [], [], []
            for n in range(0, len(atom_seq)):
                atom = atom_seq[n]
                predecessors = atom_seq[:n]
                featvecs.append(self.rec_featvec(atom, atoms, mol, self.ff, index_dict, predecessors))

                repl = np.ones(len(atoms), dtype=bool)
                repl[index_dict[atom]] = False
                repls.append(repl)

                targets.append(positions[index_dict[atom]])

            d={
                "targets": np.array(targets, dtype=np.float32),
                "positions": np.array(positions, dtype=np.float32),
                "featvec": np.array(featvecs, dtype=np.float32),
                "bond_ndx": np.array(bond_ndx, dtype=np.int64),
                "angle_ndx": np.array(ang_ndx, dtype=np.int64),
                "dih_ndx": np.array(dih_ndx, dtype=np.int64),
                "lj_ndx": np.array(lj_ndx, dtype=np.int64),
                "mol": mol,
                "repl": np.array(repls, dtype=np.bool),
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

    def rec_featvec(self, atom, atoms, mol, ff, index_dict, predecessors):
        atom_featvec = np.zeros((len(atoms), ff.n_channels))
        for index in range(0, len(atoms)):
            if atoms[index].type.channel >= 0:
                atom_featvec[index, atoms[index].type.channel] = 1
        for bond in mol.bonds:
            bond_atoms = [a for a in bond.atoms if a in predecessors]
            if bond.type.channel >= 0 and len(bond_atoms) == 1:
                for a in bond_atoms:
                    atom_featvec[index_dict[a], bond.type.channel] = 1
        for angle in mol.angles:
            angle_atoms = [a for a in angle.atoms if a in predecessors]
            if angle.type.channel >= 0 and len(angle_atoms) == 1:
                for a in angle_atoms:
                    atom_featvec[index_dict[a], angle.type.channel] = 1
        for dih in mol.dihs:
            dih_atoms = [a for a in dih.atoms if a in predecessors]
            if dih.type.channel >= 0 and len(dih_atoms) == 1:
                for a in dih_atoms:
                    atom_featvec[index_dict[a], dih.type.channel] = 1
        for lj in mol.ljs:
            lj_atoms = lj.atoms
            if lj.type.channel >= 0:
                for a in lj_atoms:
                    atom_featvec[index_dict[a], lj.type.channel] = 1
        atom_featvec[index_dict[atom], :] = 0
        return atom_featvec


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