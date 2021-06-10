import numpy as np

class Box():

    def __init__(self, file, cutoff):

        self.dim = self.get_box_dim(file)
        self.dim_inv= np.linalg.inv(self.dim)
        self.v1 = self.dim[:, 0]
        self.v2 = self.dim[:, 1]
        self.v3 = self.dim[:, 2]
        self.volume = self.get_vol()
        self.center = 0.5*self.v1 + 0.5*self.v2 + 0.5*self.v3

        self.max_v1_sub = int(self.v1[0] / cutoff)
        self.max_v2_sub = int(self.v2[1] / cutoff)
        self.max_v3_sub = int(self.v3[2] / cutoff)
        self.subbox_size = self.get_subbox_size(cutoff)
        self.subbox_size_inv = np.linalg.inv(self.subbox_size)

        #print(self.max_v1_sub)
        #print(self.max_v2_sub)
        #print(self.max_v3_sub)

    def get_box_dim(self, file):
        # reads the box dimensions from the last line in the gro file
        f_read = open(file, "r")
        bd = np.array(f_read.readlines()[-1].split(), np.float32)
        f_read.close()
        bd = list(bd)
        for n in range(len(bd), 10):
            bd.append(0.0)
        dim = np.array([[bd[0], bd[5], bd[7]],
                                 [bd[3], bd[1], bd[8]],
                                 [bd[4], bd[6], bd[2]]])
        return dim

    def get_subbox_size(self, cutoff):
        v1_scaled = self.v1
        v1_scaled = v1_scaled / self.max_v1_sub
        v2_scaled = self.v2
        v2_scaled = v2_scaled / self.max_v2_sub
        v3_scaled = self.v3
        v3_scaled = v3_scaled / self.max_v3_sub
        subbox = np.array([[v1_scaled[0], v2_scaled[0], v3_scaled[0]],
                            [v1_scaled[1], v2_scaled[1], v3_scaled[1]],
                            [v1_scaled[2], v2_scaled[2], v3_scaled[2]]])
        return subbox

    def subbox(self, pos):
        f = np.dot(self.subbox_size_inv, pos)
        f = f.astype(int)
        return tuple(f)

    def subbox_range(self, num, max):
        if num == max:
            range = [num-1, num, 0]
        elif num == 0:
            range = [max, num, num+1]
        else:
            range = [num-1, num, num+1]
        return range

    def nn_subboxes(self, sb):
        subboxes = []

        for a in self.subbox_range(sb[0], self.max_v1_sub-1):
            for b in self.subbox_range(sb[1], self.max_v2_sub-1):
                for c in self.subbox_range(sb[2], self.max_v3_sub-1):
                    subboxes.append((a,b,c))
        return list(set(subboxes))

    def empty_subbox_dict(self):
        keys = []
        for a in range(0, self.max_v1_sub):
            for b in range(0, self.max_v2_sub):
                for c in range(0, self.max_v3_sub):
                    keys.append((a,b,c))
        subbox_dict = dict([(key, []) for key in keys])
        return subbox_dict


    def move_inside(self, pos):
        f = np.dot(self.dim_inv, pos)
        g = f - np.floor(f)
        new_pos = np.dot(self.dim, g)
        return new_pos

    def diff_vec(self, diff_vec):
        diff_vec = diff_vec + self.center
        diff_vec = self.move_inside(diff_vec)
        diff_vec = diff_vec - self.center
        return diff_vec

    def diff_vec_batch(self, diff_vec):
        diff_vec = np.swapaxes(diff_vec, 0, 1)
        diff_vec = diff_vec + self.center[:, np.newaxis]
        diff_vec = self.move_inside(diff_vec)
        diff_vec = diff_vec - self.center[:, np.newaxis]
        diff_vec = np.swapaxes(diff_vec, 0, 1)
        return diff_vec

    def get_vol(self):
        norm1 = np.sqrt(np.sum(np.square(self.v1)))
        norm2 = np.sqrt(np.sum(np.square(self.v2)))
        norm3 = np.sqrt(np.sum(np.square(self.v3)))

        cos1 = np.sum(self.v2 * self.v3) / (norm2 * norm3)
        cos2 = np.sum(self.v1 * self.v3) / (norm1 * norm3)
        cos3 = np.sum(self.v1 * self.v2) / (norm1 * norm2)
        v = norm1*norm2*norm3 * np.sqrt(1-np.square(cos1)-np.square(cos2)-np.square(cos3)+2*np.sqrt(cos1*cos2*cos3))
        return v


