import os
import sys
import argparse
import numpy as np
import os.path as osp
import configparser
import torch.cuda
import set_paths
import matplotlib.pyplot as plt

from models.posenet import PoseNet, MapNet
from common.train import load_state_dict, step_feedfwd
from common.pose_utils import quaternion_angular_error, qexp
from dataset_loaders.composite import MF
from dataset_loaders.env import Env
from torch.utils.data import DataLoader
from torchvision import transforms, models

parser = argparse.ArgumentParser(description='Evaluation script for PoseNet and MapNet variants')
parser.add_argument('--dataset', type=str, default='Env', help='Dataset')
parser.add_argument('--scene', type=str, default='Lab', help='Scene name')
parser.add_argument('--weights', type=str, help='trained weights to load')
parser.add_argument('--model', type=str, choices=('posenet', 'mapnet'), help='Model to use')
parser.add_argument('--config_file', type=str, help='configuration file')
parser.add_argument('--val', action='store_true', help='Plot graph for val')
parser.add_argument('--reduce', type=int, help='Reduce training dataset')
args = parser.parse_args()

settings = configparser.ConfigParser()
with open(args.config_file, 'r') as f:
    settings.read_file(f)
seed = settings.getint('training', 'seed')
section = settings['hyperparameters']
dropout = section.getfloat('dropout')

if args.model.find('mapnet') >= 0:
    steps = section.getint('steps')
    skip = section.getint('skip')

# model
feature_extractor = models.resnet34(pretrained=False)
posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=False)

if args.model.find('mapnet') >= 0:
    model = MapNet(mapnet=posenet)
else:
    model = posenet
model.eval()

# loss functions
t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
q_criterion = quaternion_angular_error

# load weights
weights_filename = osp.expanduser(args.weights)
if osp.isfile(weights_filename):
    loc_func = lambda storage, loc: storage
    checkpoint = torch.load(weights_filename, map_location=loc_func)
    load_state_dict(model, checkpoint['model_state_dict'])
    print 'Loaded weights from {:s}'.format(weights_filename)
else:
    print 'Could not load weights from {:s}'.format(weights_filename)
    sys.exit(-1)

data_dir = osp.join('..', 'data', args.dataset)
stats_filename = osp.join(data_dir, args.scene, 'stats.txt')
stats = np.loadtxt(stats_filename)

# transformer
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))])
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# read mean and stdev for un-normalizing predictions
pose_stats_file = osp.join(data_dir, args.scene, 'pose_stats.txt')
pose_m, pose_s  = np.loadtxt(pose_stats_file)  # max value

# dataset
train = not args.val
if train:
    print 'Running {:s} on Training Dataset'.format(args.model)
else:
    print 'Running {:s} on Validation Dataset'.format(args.model)

data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
kwargs = dict(dataset=args.dataset, scene=args.scene, data_path=data_dir, train=train, transform=data_transform, target_transform=target_transform, reduce=args.reduce, seed=seed)

if args.model.find('mapnet') >= 0:
    data_set = MF(steps=steps, skip=skip, **kwargs)
    L = len(data_set.dset)
else:
    data_set = Env(**kwargs)
    L = len(data_set)

# loader (batch_size MUST be 1)
batch_size = 1
assert batch_size == 1
loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)

# activate GPUs
CUDA = torch.cuda.is_available()
torch.manual_seed(seed)
if CUDA:
    torch.cuda.manual_seed(seed)
    model.cuda()

pred_poses = np.zeros((L, 7))  # store all predicted poses
targ_poses = np.zeros((L, 7))  # store all target poses

# inference loop
for batch_idx, (data, target) in enumerate(loader):
    if batch_idx % 200 == 0:
        print 'Image {:d} / {:d}'.format(batch_idx, len(loader))

    idx = [batch_idx]
    idx = idx[len(idx) / 2]

    # output : 1 x 6 or 1 x STEPS x 6
    _, output = step_feedfwd(data, model, CUDA, train=False)
    s = output.size()
    output = output.cpu().data.numpy().reshape((-1, s[-1]))
    target = target.numpy().reshape((-1, s[-1]))

    # normalize the predicted quaternions
    q = [qexp(p[3:]) for p in output]
    output = np.hstack((output[:, :3], np.asarray(q)))
    q = [qexp(p[3:]) for p in target]
    target = np.hstack((target[:, :3], np.asarray(q)))

    # un-normalize the predicted and target translations
    output[:, :3] = (output[:, :3] * pose_s) + pose_m
    target[:, :3] = (target[:, :3] * pose_s) + pose_m

    # pred_poses[idx, :] = output[len(output) / 2]
    # targ_poses[idx, :] = target[len(target) / 2]
    pred_poses[idx, :] = output[-1]
    targ_poses[idx, :] = target[-1]


# calculate losses
t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, :3], targ_poses[:, :3])])
q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:], targ_poses[:, 3:])])

print 'Error in translation: median {:3.2f} m,  mean {:3.2f} m\nError in rotation: median {:3.2f} degrees, mean {:3.2f} degree'.\
    format(np.median(t_loss), np.mean(t_loss), np.median(q_loss), np.mean(q_loss))

# create figure object
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

# plot on the figure object
ss = 2
# scatter the points and draw connecting line
x = np.vstack((pred_poses[::ss, 0].T, targ_poses[::ss, 0].T))
y = np.vstack((pred_poses[::ss, 1].T, targ_poses[::ss, 1].T))

# 2D drawing
ax.plot(x, y, lw=0.1, c='b')
ax.scatter(x[0, :], y[0, :], s=3, c='r')
ax.scatter(x[1, :], y[1, :], s=3, c='g')

ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show(block=True)



