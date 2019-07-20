"""
Computes the mean and std of pixels in a dataset
"""
import os.path as osp
import numpy as np
import argparse
import set_paths
from dataset_loaders.env import Env

from torchvision import transforms
from torch.utils.data import DataLoader
from common.train import safe_collate

parser = argparse.ArgumentParser(description='Dataset images statistics')
parser.add_argument('--dataset', type=str, help='Dataset', required=True)
parser.add_argument('--scene', type=str, help='Scene name', required=True)
args = parser.parse_args()

data_transform = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor()])

# dataset loader
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
kwargs = dict(dataset= args.dataset, scene=args.scene, data_path=data_dir, train=True, transform=data_transform)
dset = Env(**kwargs)

# accumulate
batch_size = 100
num_workers = batch_size
loader = DataLoader(dset, batch_size=batch_size, num_workers=num_workers, collate_fn=safe_collate)
acc = np.zeros((3, 224, 224))
sq_acc = np.zeros((3, 224, 224))

for batch_idx, (imgs, _) in enumerate(loader):
    imgs = imgs.numpy()
    acc += np.sum(imgs, axis=0)
    sq_acc += np.sum(imgs ** 2, axis=0)

    if batch_idx % 10 == 0:
        print 'Accumulated {:d} / {:d}'.format(batch_idx * batch_size, len(dset))

N = len(dset) * acc.shape[1] * acc.shape[2]

mean_p = np.asarray([np.sum(acc[c]) for c in xrange(3)])
mean_p /= N
print 'Mean pixel = ', mean_p

# std = E[x^2] - E[x]^2
std_p = np.asarray([np.sum(sq_acc[c]) for c in xrange(3)])
std_p /= N
std_p -= (mean_p ** 2)
print 'Std. pixel = ', std_p

output_filename = osp.join('..', 'data', args.dataset, args.scene, 'stats.txt')
np.savetxt(output_filename, np.vstack((mean_p, std_p)), fmt='%8.7f')
print '{:s} written'.format(output_filename)