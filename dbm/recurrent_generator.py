import numpy as np
import math

class Recurrent_Generator():

    def __init__(self, data, train=False, hydrogens=False, gibbs=False, rand_rot=False, pad_seq=True, ref_pos=True):

        self.data = data
        self.train = train
        self.hydrogens = hydrogens
        self.gibbs = gibbs
        self.rand_rot = rand_rot
        self.pad_seq = pad_seq
        self.ref_pos = ref_pos

        if train:
            self.samples = self.data.samples_train
        else:
            self.samples = self.data.samples_val

    def __iter__(self):

        #go through every sample
        for sample in self.samples:
            #get sequence of beads to visit
            bead_seq = sample.gen_bead_seq(train=self.train)
            #choose dict for atoms in a givne bead (heavy or hydrogens)
            if self.hydrogens:
                atom_seq_dict = sample.aa_seq_hydrogens
            else:
                atom_seq_dict = sample.aa_seq_heavy
            #visit every bead
            for bead in bead_seq:
                d = {}
                #start_atom = atom_seq_dict[bead][0]
                cg_f = sample.cg_features[bead]
                loc_env = sample.loc_envs[bead]

                #CG features
                d["cg_feat"] = np.array(self.pad2d(cg_f.fv, self.data.max['beads_loc_env']), dtype=np.float32)
                d["cg_pos"] = np.array(self.pad2d(loc_env.bead_positions(), self.data.max['beads_loc_env']), dtype=np.float32)

                #env atom positions
                if self.ref_pos:
                    d["aa_pos"] = self.pad2d(loc_env.atom_positions_ref(), self.data.max['atoms_loc_env'])
                else:
                    d["aa_pos"] = self.pad2d(loc_env.atom_positions(), self.data.max['atoms_loc_env'])

                #just for debugging...
                #d["aa_pos_ref"] = self.pad2d(loc_env.atom_positions_ref(), self.data.max['atoms_loc_env'])

                if not atom_seq_dict[bead]:
                    continue

                target_pos, target_type, aa_feat, repl = [], [], [], []
                bonds_ndx, angles_ndx, dihs_ndx, ljs_ndx = [], [], [], []
                for atom in atom_seq_dict[bead]:
                    aa_f = sample.aa_features[atom]

                    #target position
                    t_pos = loc_env.rot(np.array([atom.ref_pos]))
                    target_pos.append(t_pos)

                    #target atom type
                    t_type = np.zeros(self.data.ff.n_atom_chns)
                    t_type[atom.type.index] = 1

                    #t_type[atom.type.channel] = 1
                    target_type.append(t_type)

                    if self.gibbs:
                        atom_featvec = self.pad2d(aa_f.fv_gibbs, self.data.max['atoms_loc_env'])
                        bonds_ndx += aa_f.energy_ndx_gibbs['bonds']
                        angles_ndx += aa_f.energy_ndx_gibbs['angles']
                        dihs_ndx += aa_f.energy_ndx_gibbs['dihs']
                        ljs_ndx += aa_f.energy_ndx_gibbs['ljs']
                    else:
                        atom_featvec = self.pad2d(aa_f.fv_init, self.data.max['atoms_loc_env'])
                        bonds_ndx += aa_f.energy_ndx_init['bonds']
                        angles_ndx += aa_f.energy_ndx_init['angles']
                        dihs_ndx += aa_f.energy_ndx_init['dihs']
                        ljs_ndx += aa_f.energy_ndx_init['ljs']

                    #AA featurevector
                    aa_feat.append(atom_featvec)

                    #replace vector: marks the index of the target atom in "aa_pos" (needed for recurrent training)
                    r = self.pad1d(aa_f.repl, self.data.max['atoms_loc_env'], value=True)
                    repl.append(r)

                #pad energy terms
                d["bonds_ndx"] = np.array(self.pad_energy_ndx(bonds_ndx, self.data.max['bonds_per_bead']), dtype=np.int64)
                d["angles_ndx"] = np.array(self.pad_energy_ndx(angles_ndx, self.data.max['angles_per_bead'], tuple([-1, 1, 2, 3])), dtype=np.int64)
                d["dihs_ndx"] = np.array(self.pad_energy_ndx(dihs_ndx, self.data.max['dihs_per_bead'], tuple([-1, 1, 2, 3, 4])), dtype=np.int64)
                d["ljs_ndx"] = np.array(self.pad_energy_ndx(ljs_ndx, self.data.max['ljs_per_bead']), dtype=np.int64)

                # Padding for recurrent training
                if self.pad_seq:
                    for n in range(0, self.data.max['seq_len'] - len(atom_seq_dict[bead])):
                        target_pos.append(np.zeros((1, 3)))
                        target_type.append(target_type[-1])
                        aa_feat.append(np.zeros(aa_feat[-1].shape))
                        repl.append(np.ones(repl[-1].shape, dtype=bool))
                d["target_pos"] = np.array(target_pos, dtype=np.float32)
                d["target_type"] = np.array(target_type, dtype=np.float32)
                d["aa_feat"] = np.array(aa_feat, dtype=np.float32)
                d["repl"] = np.array(repl, dtype=np.bool)

                #mask for sequences < max_seq_len
                mask = np.zeros(self.data.max['seq_len'])
                mask[:len(atom_seq_dict[bead])] = 1
                d["mask"] = np.array(mask, dtype=np.float32)

                if self.rand_rot:
                    rot_mat = self.rand_rot_mat()
                    d["target_pos"] = np.dot(d["target_pos"], rot_mat)
                    d["aa_pos"] = np.dot(d["aa_pos"], rot_mat)
                    d["cg_pos"] = np.dot(d["cg_pos"], rot_mat)

                d['loc_env'] = loc_env
                d['atom_seq'] = atom_seq_dict[bead]

                yield d

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