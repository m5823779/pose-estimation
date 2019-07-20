"""
Main training script for MapNet
"""
import json
import torch
import argparse
import set_paths
import numpy as np
import os.path as osp
import configparser

from torch import nn
from torchvision import transforms, models
from common.train import Trainer
from common.optimizer import Optimizer
from common.criterion import PoseNetCriterion, MapNetCriterion, MapNetOnlineCriterion
from models.posenet import PoseNet, MapNet
from dataset_loaders.composite import MF
from dataset_loaders.env import Env

parser = argparse.ArgumentParser(description='Training script for PoseNet and MapNet variants')
parser.add_argument('--dataset', type=str, default='Env', help='Dataset')
parser.add_argument('--scene', type=str, default='Lab', help='Scene name')
parser.add_argument('--config_file', type=str, help='configuration file')
parser.add_argument('--model', choices=('posenet', 'mapnet'), help='Model to train')
parser.add_argument('--checkpoint', type=str, help='Checkpoint to resume from', default=None)
parser.add_argument('--reduce', type=int, help='Reduce training dataset')
args = parser.parse_args()

settings = configparser.ConfigParser()
with open(args.config_file, 'r') as f:
  settings.read_file(f)
section = settings['optimization']
optim_config = {k: json.loads(v) for k,v in section.items() if k != 'opt'}
opt_method = section['opt']
lr = optim_config.pop('lr')
weight_decay = optim_config.pop('weight_decay')

section = settings['hyperparameters']
dropout = section.getfloat('dropout')
color_jitter = section.getfloat('color_jitter', 0)
sax = 0.0
saq = section.getfloat('beta')

if args.model.find('mapnet') >= 0:
  skip = section.getint('skip')
  steps = section.getint('steps')
  srx = 0.0
  srq = section.getfloat('gamma')

section = settings['training']
seed = section.getint('seed')

# model
feature_extractor = models.resnet34(pretrained=True)
posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=True, filter_nans=False)

if args.model == 'posenet':
  model = posenet
elif args.model.find('mapnet') >= 0:
  model = MapNet(mapnet=posenet)
else:
  raise NotImplementedError

# loss function
if args.model == 'posenet':
  train_criterion = PoseNetCriterion(sax=sax, saq=saq, learn_beta=True)
  val_criterion = PoseNetCriterion()
elif args.model.find('mapnet') >= 0:
  kwargs = dict(sax=sax, saq=saq, srx=srx, srq=srq, learn_beta=True, learn_gamma=True)
  train_criterion = MapNetCriterion(**kwargs)
  val_criterion = MapNetCriterion()
else:
  raise NotImplementedError

# optimizer
param_list = [{'params': model.parameters()}]
if hasattr(train_criterion, 'sax') and hasattr(train_criterion, 'saq'):
  param_list.append({'params': [train_criterion.sax, train_criterion.saq]})
if hasattr(train_criterion, 'srx') and hasattr(train_criterion, 'srq'):
  param_list.append({'params': [train_criterion.srx, train_criterion.srq]})
optimizer = Optimizer(params=param_list, method=opt_method, base_lr=lr, weight_decay=weight_decay, **optim_config)

data_dir = osp.join('..', 'data', args.dataset)
stats_file = osp.join(data_dir, args.scene, 'stats.txt')
stats = np.loadtxt(stats_file)

# transformers
tforms = [transforms.Resize(256)]
if color_jitter > 0:
  assert color_jitter <= 1.0
  print 'Using ColorJitter data augmentation'
  tforms.append(transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter, hue=0.5))
tforms.append(transforms.ToTensor())
tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
data_transform = transforms.Compose(tforms)
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# datasets
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
kwargs = dict(dataset=args.dataset, scene=args.scene, data_path=data_dir, transform=data_transform, target_transform=target_transform, reduce=args.reduce, seed=seed)

if args.model == 'posenet':
  train_set = Env(train=True, **kwargs)
  val_set = Env(train=False, **kwargs)
elif args.model.find('mapnet') >= 0:
  kwargs = dict(kwargs, skip=skip, steps=steps)
  train_set = MF(train=True, **kwargs)
  val_set = MF(train=False, **kwargs)
else:
  raise NotImplementedError

# trainer
config_name = args.config_file.split('/')[-1]
config_name = config_name.split('.')[0]
if args.reduce is None:
  experiment_name = '{:s}_{:s}_{:s}_{:s}'.format(args.dataset, args.scene, args.model, config_name)
else:
  experiment_name = '{:s}_{:s}_{:s}_{:s}_reduce'.format(args.dataset, args.scene, args.model, config_name)
trainer = Trainer(model, optimizer, train_criterion, args.config_file,
                  experiment_name, train_set, val_set, device='0', checkpoint_file=args.checkpoint,
                  resume_optim=False, val_criterion=val_criterion)
trainer.train_val()
