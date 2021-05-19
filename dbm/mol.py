import numpy as np
import os
from os import path
import pickle
import mdtraj as md
import networkx as nx
# from tqdm.auto import tqdm
from timeit import default_timer as timer
import itertools
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from dbm.ff import *
from dbm.util import read_between

elem_dict = {
    "H_AR": "H",
    "H": "H",
    "C_AR": "C",
    "C": "C"}

class Atom():

    index = 0
    mol_index = 0
    def __init__(self, mol, type, subbox, pos=None):
        self.index = Atom.index
        Atom.index += 1
        self.mol_index = Atom.mol_index
        Atom.mol_index += 1
        self.mol = mol
        self.type = type
        if pos is None:
            self.pos = np.zeros(3)
        else:
            self.pos = pos
        self.subbox = subbox

class Bead():
    index = 0

    def __init__(self, mol, type, pos=None, fp=None, mult=1):
        self.index = Bead.index
        Bead.index += 1
        self.mol = mol
        if pos is None:
            self.pos = np.zeros(3)
        else:
            self.pos = pos
        self.type = type

        self.fp = fp
        self.mult = mult



class Mol():

    index = 0
    def __init__(self, name, box, ff, beads = None, atoms = None, top_file = None):
        self.name = name
        self.index = Mol.index
        Mol.index += 1
        if beads is None:
            self.beads = []
        else:
            self.beads = beads
        if atoms is None:
            self.atoms = []
        else:
            self.atoms = atoms

        self.G = None
        self.G_heavy = None
        self.hydrogens = None

        self.bonds = []
        self.angles = []
        self.dihs = []
        self.excls = []
        self.pairs = []

        self.cg_edges = []

        self.fp = {}

        self.top_file = top_file
        self.ff = ff

        self.com = None
        self.box = box
        self.env_mols = None
        self.intermolecular_atoms = None
        self.intermolecular_beads = None

    def compute_com(self):
        com, mass, = np.zeros(3), 0.0
        for a in self.atoms:
            # center on first position to get a whole molecule over pbc (assuming mol < box_len)
            pos = self.box.diff_vec(a.pos - self.atoms[0].pos)
            com += pos * a.type.mass
            mass += a.type.mass
        self.com = self.box.move_inside(com / mass + self.atoms[0].pos)

    def add_bead(self, bead):
        self.beads.append(bead)

    def add_atom(self, atom):
        self.atoms.append(atom)

    def add_aa_top(self, top_file):

        self.top_file = top_file

        for line in read_between("[bonds]", "[/bonds]", top_file):
            if len(line.split()) >= 2:
                index1 = int(line.split()[0]) - 1
                index2 = int(line.split()[1]) - 1
                bond = self.ff.make_bond([self.atoms[index1], self.atoms[index2]])
                if bond:
                    self.add_bond(bond)

        for line in read_between("[angles]", "[/angles]", top_file):
            if len(line.split()) >= 3:
                index1 = int(line.split()[0]) - 1
                index2 = int(line.split()[1]) - 1
                index3 = int(line.split()[2]) - 1
                angle = self.ff.make_angle([self.atoms[index1], self.atoms[index2], self.atoms[index3]])
                if angle:
                    self.add_angle(angle)

        for line in read_between("[dihedrals]", "[/dihedrals]", top_file):
            if len(line.split()) >= 4:
                index1 = int(line.split()[0]) - 1
                index2 = int(line.split()[1]) - 1
                index3 = int(line.split()[2]) - 1
                index4 = int(line.split()[3]) - 1
                dih = self.ff.make_dih([self.atoms[index1], self.atoms[index2], self.atoms[index3], self.atoms[index4]])
                if dih:
                    self.add_dih(dih)

        for line in read_between("[exclusions]", "[/exclusions]", top_file):
            if len(line.split()) >= 2:
                index1 = int(line.split()[0]) - 1
                index2 = int(line.split()[1]) - 1
                self.add_excl((self.atoms[index1], self.atoms[index2]))

        for line in read_between("[pairs]", "[/pairs]", top_file):
            if len(line.split()) >= 2:
                index1 = int(line.split()[0]) - 1
                index2 = int(line.split()[1]) - 1
                self.add_pair((self.atoms[index1], self.atoms[index2]))

        self.make_aa_graph()

    def make_ljs(self):
        #self.intermolecular_atoms = intermolecular_atoms
        #nexcl_atoms: bonded atoms up to n_excl
        #lengths, paths = nx.multi_source_dijkstra(self.G, self.atoms, cutoff=ff.n_excl)
        #lengths, paths = nx.all_pairs_dijkstra(self.G, self.atoms, cutoff=ff.n_excl)
        paths = nx.all_pairs_shortest_path(self.G, cutoff=self.ff.n_excl)
        path_exclusions = []
        for p in paths:
            path_exclusions += [(t[0], t[-1]) for t in list(p[1].values())]
            #print("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
            #print([(t[0], t[-1]) for t in list(p[1].values())])
        path_exclusions = list(set(path_exclusions))
        #print(len(path_exclusions))
        #exclusions = [(t[0], t[1][0]) for t in list(paths.items())] + self.excls
        #print([list(p[1].values()) for p in list(paths)])
        #print(list(set(itertools.chain.from_iterable([p[1].values() for p in paths]))))
        #exclusions = [(t[0], t[-1]) for t in list(paths)] + self.excls
        #print(exclusions)
        #print(lengths)
        exclusions = path_exclusions + self.excls

        #generate intra molecular lj pairs:
        lj_pairs_intra = []
        for n in range(0, len(self.atoms)):
            for m in range(n+1, len(self.atoms)):
                lj_pairs_intra.append((self.atoms[n], self.atoms[m]))
        #remove exclusions and duplicates and add
        #print("1", len(lj_pairs_intra))
        #print(lj_pairs_intra)
        #print(exclusions)
        #a = [(t[0].mol_index, t[1].mol_index) for t in lj_pairs_intra]
        #b = [(t[0].mol_index, t[1].mol_index) for t in path_exclusions]
        #print(b)

        #print(a)
        #print(b)
        #for e in b:
        #    if e in a:
        #        print("hie rist es drinne!!!")
        lj_pairs_intra = list(set(lj_pairs_intra) - set(exclusions))
        #print("2", len(lj_pairs_intra))
        exclusions_reversed = [(e[1], e[0]) for e in exclusions]
        lj_pairs_intra = list(set(lj_pairs_intra) - set(exclusions_reversed))
        #print("3", len(lj_pairs_intra))

        #add pairs
        lj_pairs_intra = list(set(lj_pairs_intra + self.pairs))
        #print("4", len(lj_pairs_intra))

        #remove reversed duplicates
        for p in lj_pairs_intra:
            if (p[1], p[0]) in lj_pairs_intra:
                lj_pairs_intra.remove((p[1], p[0]))
        #print("5", len(lj_pairs_intra))

        #generate inter molecular lj pairs
        lj_pairs_inter = []
        for a in self.atoms:
            for b in self.intermolecular_atoms:
                lj_pairs_inter.append((a, b))

        #all lj pairs
        lj_pairs = lj_pairs_intra + lj_pairs_inter

        #equip with ff parameters
        self.ljs_intra = self.ff.make_ljs(lj_pairs_intra)
        self.ljs = self.ff.make_ljs(lj_pairs)

    def add_cg_top(self, top_file):

        for line in read_between("[bonds]", "[/bonds]", top_file):
            index1 = int(line.split()[0]) - 1
            index2 = int(line.split()[1]) - 1
            self.add_cg_edge([self.beads[index1], self.beads[index2]])

        self.make_cg_graph()

    def add_bond(self, bond):
        self.bonds.append(bond)

    def add_angle(self, angle):
        self.angles.append(angle)

    def add_dih(self, dih):
        self.dihs.append(dih)

    def add_excl(self, excl):
        self.excls.append(excl)

    def add_pair(self, pair):
        self.pairs.append(pair)

    def add_cg_edge(self, edge):
        self.cg_edges.append(edge)

    def make_preference_axis(self):

        if not self.top_file.exists():
            raise Exception('No topology file. Add topology file before calling make_preference_axis')

        align = None
        for line in read_between("[align]", "[/align]", self.top_file):
            ndx1, ndx2 = line.split()
            align = (int(ndx1)-1, int(ndx2)-1)

        if align:
            c1 = self.atoms[align[0]]
            c2 = self.atoms[align[1]]

            # compute rotation matrix to align loc env (aligns fixpoint vector with z-axis)
            v1 = np.array([0.0, 0.0, 1.0])
            v2 = self.box.diff_vec(c1.pos - c2.pos)

            # rotation axis
            v_rot = np.cross(v1, v2)
            v_rot = v_rot / np.linalg.norm(v_rot)

            # rotation angle
            cosang = np.dot(v1, v2)
            sinang = np.linalg.norm(np.cross(v1, v2))
            theta = np.arctan2(sinang, cosang)

            # rotation matrix
            a = math.cos(theta / 2.0)
            b, c, d = -v_rot * math.sin(theta / 2.0)
            aa, bb, cc, dd = a * a, b * b, c * c, d * d
            bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
            rot_mat = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

            self.rot_mat = rot_mat
        else:
            self.rot_mat = np.identity(3)



    def make_aa_graph(self):
        self.G = nx.Graph()
        self.G.add_nodes_from(self.atoms)
        edges = [bond.atoms for bond in self.bonds]
        self.G.add_edges_from(edges)

        heavy_atoms = [a for a in self.atoms if a.type.mass >= 2.0]
        heavy_edges = [e for e in edges if e[0].type.mass >= 2.0 and e[1].type.mass >= 2.0]
        self.G_heavy = nx.Graph()
        self.G_heavy.add_nodes_from(heavy_atoms)
        self.G_heavy.add_edges_from(heavy_edges)

        self.hydrogens = [a for a in self.atoms if a.type.mass < 2.0]

    def make_cg_graph(self):
        self.G_cg = nx.Graph()
        self.G_cg.add_nodes_from(self.beads)
        self.G_cg.add_edges_from(self.cg_edges)

    def cg_seq(self, order="dfs", train=True):

        #if order == "dfs":
        #    beads = list(nx.dfs_preorder_nodes(self.G_cg))
        #breath first search
        if order == "bfs":
            edges = list(nx.bfs_edges(self.G_cg, np.random.choice(self.beads)))
            beads = [edges[0][0]] + [e[1] for e in edges]
        #random search
        elif order == "random":
            beads = [np.random.choice(self.beads)]
            pool = []
            for n in range(1, len(self.beads)):
                pool += list(nx.neighbors(self.G_cg, beads[-1]))
                pool = list(set(pool))
                next = np.random.choice(pool)
                while next in beads:
                    next = np.random.choice(pool)
                pool.remove(next)
                beads.append(next)
        # depth first search (default)
        else:
            beads = list(nx.dfs_preorder_nodes(self.G_cg))

        # data augmentation for undersampled beads
        seq = []
        for n in range(0, len(beads)):
            if train:
                seq += [(beads[n], beads[:n])]*beads[n].mult
            else:
                seq.append((beads[n], beads[:n]))

        # shuffle sequence if training
        if train:
            np.random.shuffle(seq)

        return seq

    def aa_seq(self, order="dfs", train=True):
        mol_atoms_heavy = [a for a in self.atoms if a.type.mass >= 2.0]
        atom_seq_dict_heavy = {}
        atom_seq_dict_hydrogens = {}
        atom_predecessors_dict = {}
        cg_seq = self.cg_seq(order=order, train=train)
        for bead, predecessor_beads in cg_seq:
            bead_atoms = bead.atoms
            heavy_atoms = [a for a in bead_atoms if a.type.mass >= 2.0]
            hydrogens = [a for a in bead_atoms if a.type.mass < 2.0]
            predecessor_atoms = list(itertools.chain.from_iterable([b.atoms for b in set(predecessor_beads)]))
            predecessor_atoms_heavy = [a for a in predecessor_atoms if a.type.mass >= 2.0]
            predecessor_atoms_hydrogens = [a for a in predecessor_atoms if a.type.mass < 2.0]

            #find start atom
            psble_start_nodes = []
            n_heavy_neighbors = []
            for a in heavy_atoms:
                n_heavy_neighbors.append(len(list(nx.all_neighbors(self.G_heavy, a))))
                for n in nx.all_neighbors(self.G_heavy, a):
                    if n in predecessor_atoms_heavy:
                        psble_start_nodes.append(a)
            if psble_start_nodes:
                #start_atom = np.random.choice(psble_start_nodes)
                #weird bonds in cg sPS... therefore just take first one...
                start_atom = psble_start_nodes[0]
            else:
                start_atom = heavy_atoms[np.array(n_heavy_neighbors).argmin()]
            #else:
            #    start_atom = heavy_atoms[0]

            #sequence through atoms of bead
            if order == "bfs":
                edges = list(nx.bfs_edges(self.G.subgraph(heavy_atoms), start_atom))
                atom_seq = [start_atom] + [e[1] for e in edges]
            elif order == "random":
                atom_seq = [start_atom]
                pool = []
                for n in range(1, len(heavy_atoms)):
                    pool += list(nx.neighbors(self.G.subgraph(heavy_atoms), atom_seq[-1]))
                    pool = list(set(pool))
                    next = np.random.choice(pool)
                    while next in atom_seq:
                        next = np.random.choice(pool)
                    pool.remove(next)
                    atom_seq.append(next)
            else:
                atom_seq = list(nx.dfs_preorder_nodes(self.G.subgraph(heavy_atoms), start_atom))
            #hydrogens = self.hydrogens[:]
            np.random.shuffle(hydrogens)
            #atom_seq = atom_seq + hydrogens

            #atom_seq = []
            for n in range(0, len(atom_seq)):
                atom_predecessors_dict[atom_seq[n]] = predecessor_atoms_heavy + atom_seq[:n]
            for n in range(0, len(hydrogens)):
                atom_predecessors_dict[hydrogens[n]] = mol_atoms_heavy + predecessor_atoms_hydrogens + hydrogens[:n]

            atom_seq_dict_heavy[bead] = atom_seq
            atom_seq_dict_hydrogens[bead] = hydrogens


        return cg_seq, atom_seq_dict_heavy, atom_seq_dict_hydrogens, atom_predecessors_dict


