"""
pytorch data loader for the dataset
"""
import os
import os.path as osp
import numpy as np
from torch.utils import data
from utils import load_image


class Env(data.Dataset):
    def __init__(self, dataset, scene, data_path, train, transform=None, target_transform=None, reduce=None, seed=7):
        self.transform = transform
        self.target_transform = target_transform
        self.reduce = reduce
        np.random.seed(seed)

        # directories
        base_dir = osp.join(osp.expanduser(data_path), scene)
        data_dir = osp.join('..', 'data', dataset, scene)

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # decide which sequences to use
        if train:
            split_file = osp.join(base_dir, 'TrainSplit.txt')
        else:
            split_file = osp.join(base_dir, 'TestSplit.txt')
        with open(split_file, 'r') as f:
            seqs = [int(l.split('Sequence')[-1]) for l in f if not l.startswith('#')]

        # read poses and collect image names
        self.c_imgs = []
        ps = {}
        # self.gt_idx = np.empty((0,), dtype=np.int)
        # gt_offset = int(0)

        for seq in seqs:
            seq_dir = osp.join(base_dir, 'seq-{:02d}'.format(seq))
            p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if n.find('pose') >= 0]
            frame_idx = np.array(xrange(len(p_filenames)), dtype=np.int)

            if self.reduce is None:
                pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.format(i))).flatten()[:7] for i in frame_idx]
            else:
                pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.format(i))).flatten()[:7] for i in frame_idx if i%self.reduce==0]
            ps[seq] = np.asarray(pss)

            # self.gt_idx = np.hstack((self.gt_idx, gt_offset + frame_idx))
            # gt_offset += len(p_filenames)

            if self.reduce is None:
                c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i)) for i in frame_idx]
            else:
                c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i)) for i in frame_idx if i%self.reduce==0]
            self.c_imgs.extend(c_imgs)

        # read / save pose normalization information
        poses = np.empty((0, 7))
        for p in ps.values():
            poses = np.vstack((poses, p))
        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')

        if train:
            mean_t = np.mean(poses[:, [0, 1, 2]], axis=0)
            std_t = np.std(poses[:, [0, 1, 2]], axis=0)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        # convert pose to translation + log quaternion, normalize
        self.poses = np.empty((0, 6))
        for seq in seqs:
            pss = self.process(poses_in=ps[seq], mean_t=mean_t, std_t=std_t)
            self.poses = np.vstack((self.poses, pss))

    def __getitem__(self, index):
        img = None
        while img is None:
            img = load_image(self.c_imgs[index])
            pose = self.poses[index]
            index += 1
        index -= 1

        if self.target_transform is not None:
            pose = self.target_transform(pose)

        if self.transform is not None:
            img = self.transform(img)

        return img, pose

    def process(self, poses_in, mean_t, std_t):
        poses_out = np.zeros((len(poses_in), 6))
        poses_out[:, 0:3] = poses_in[:, [0, 1, 2]]
        for i in xrange(len(poses_out)):
            q = poses_in[i, [3, 4, 5, 6]]
            q *= np.sign(q[0])  # constrain to hemisphere
            q = self.qlog(q)
            poses_out[i, 3:] = q

        # normalize translation
        poses_out[:, :3] -= mean_t
        if std_t[2] != 0:
            poses_out[:, :3] /= std_t
        else:
            poses_out[:, :2] /= std_t[:2]

        return poses_out

    def qlog(self, q):
        if all(q[1:] == 0):
            q = np.zeros(3)
        else:
            q = np.arccos(np.minimum(1, q[0])) * q[1:] / np.linalg.norm(q[1:])
        return q

    def __len__(self):
        return self.poses.shape[0]

