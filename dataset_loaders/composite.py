"""
Composite data-loaders derived from class specific data loaders
"""
import torch
import numpy as np
from torch.utils import data
from torch.autograd import Variable
from dataset_loaders.env import Env


class MF(data.Dataset):
  """
  Returns multiple consecutive frames
  """
  def __init__(self, *args, **kwargs):
    """
    :param steps: Number of frames to return on every call
    :param skip: Number of frames to skip
    """
    self.steps = kwargs.pop('steps', 2)
    self.skip = kwargs.pop('skip', 1)
    self.train = kwargs['train']
    self.dset = Env(*args, **kwargs)
    self.L = self.steps * self.skip

  def get_indices(self, index):
    skips = self.skip * np.ones(self.steps-1)  # [10.  10.]
    offsets = np.insert(skips, 0, 0).cumsum()  # [0.  10.  20.]
    offsets -= offsets[-1]  # [-20.  -10.  0.]
    offsets = offsets.astype(np.int)
    idx = index + offsets
    idx = np.minimum(np.maximum(idx, 0), len(self.dset)-1)
    assert np.all(idx >= 0), '{:d}'.format(index)
    assert np.all(idx < len(self.dset))
    return idx

  def __getitem__(self, index):
    """
    :param index: 
    :return: imgs: STEPS x 3 x H x W
             poses: STEPS x 7
    """
    idx = self.get_indices(index)
    clip = [self.dset[i] for i in idx]

    imgs = torch.stack([c[0] for c in clip], dim=0)
    poses = torch.stack([c[1] for c in clip], dim=0)

    return imgs, poses

  def __len__(self):
    L = len(self.dset)
    return L


