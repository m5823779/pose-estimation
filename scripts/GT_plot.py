import os
import os.path as osp
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Ground truth plot')
parser.add_argument('--dataset', type=str, default='Env', help='Dataset')
parser.add_argument('--scene', type=str, default='Lab', help='Scene name')
args = parser.parse_args()

# directories
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset, args.scene)


split_file = osp.join(data_dir, 'TrainSplit.txt')
with open(split_file, 'r') as f:
    seqs = [int(l.split('Sequence')[-1]) for l in f if not l.startswith('#')]

ps = {}
for seq in seqs:
    seq_dir = osp.join(data_dir, 'seq-{:02d}'.format(seq))
    p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if n.find('pose') >= 0]
    frame_idx = np.array(xrange(len(p_filenames)), dtype=np.int)
    pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.format(i))).flatten()[:7] for i in frame_idx]
    ps[seq] = np.asarray(pss)

gt_poses = np.empty((0, 7))
for p in ps.values():
    gt_poses = np.vstack((gt_poses, p))

# create figure object
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)


# scatter the points and draw connecting line
x = gt_poses[::1, 0].T
y = gt_poses[::1, 1].T

print 'Plotting Ground truth...'
# 2D drawing
ax.scatter(x[:], y[:], s=5, c='g')

ax.set_ylabel('Y')
ax.set_xlabel('X')
print 'Done!'
print 'Close windows to exit...'
plt.show(block=True)


