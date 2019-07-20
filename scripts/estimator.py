import os
import sys
import numpy as np
import os.path as osp
import configparser
import torch.cuda
import set_paths

from models.posenet import PoseNet, MapNet
from common.train import load_state_dict, step_feedfwd
from common.pose_utils import qexp
from torchvision import transforms, models

class Estimator(object):
    def __init__(self, dataset, scene, weights, model, config_file):
        self.dataset = dataset
        self.scene = scene
        self.weights = weights
        self.model = model
        self.config_file = config_file

        self.config()
        self.eval_net = self.build_network()
        self.load_network()
        self.load_stats()

    def config(self):
        settings = configparser.ConfigParser()
        with open(self.config_file, 'r') as f:
            settings.read_file(f)
        self.seed = settings.getint('training', 'seed')
        section = settings['hyperparameters']
        self.dropout = section.getfloat('dropout')

    def build_network(self):
        # model
        feature_extractor = models.resnet34(pretrained=False)
        posenet = PoseNet(feature_extractor, droprate=self.dropout, pretrained=False)

        if self.model.find('mapnet') >= 0:
            model = MapNet(mapnet=posenet)
        else:
            model = posenet

        return model.eval()

    def load_network(self):
        # load weights
        weights_filename = osp.expanduser(self.weights)
        if osp.isfile(weights_filename):
            loc_func = lambda storage, loc: storage
            checkpoint = torch.load(weights_filename, map_location=loc_func)
            load_state_dict(self.eval_net, checkpoint['model_state_dict'])
            print 'Loaded weights from {:s}'.format(weights_filename)
        else:
            print 'Could not load weights from {:s}'.format(weights_filename)
            sys.exit(-1)

    def load_stats(self):
        data_dir = osp.join('..', 'data', self.dataset)
        stats_filename = osp.join(data_dir, self.scene, 'stats.txt')
        self.stats = np.loadtxt(stats_filename)

        # read mean and stdev for un-normalizing predictions
        pose_stats_file = osp.join(data_dir, self.scene, 'pose_stats.txt')
        self.max_value = np.loadtxt(pose_stats_file)  # max value

    def transform(self, img):
        # transformer
        data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.stats[0], std=np.sqrt(self.stats[1]))])
        img = data_transform(img)

        return img

    def estimation(self, img):
        # activate GPUs
        CUDA = torch.cuda.is_available()
        torch.manual_seed(self.seed)
        if CUDA:
            torch.cuda.manual_seed(self.seed)
            self.eval_net.cuda()

        # output : 1 x 6 or 1 x STEPS x 6
        _, output = step_feedfwd(img, self.eval_net, CUDA, train=False)
        s = output.size()
        output = output.cpu().data.numpy().reshape((-1, s[-1]))

        # normalize the predicted quaternions
        q = [qexp(p[3:]) for p in output]
        output = np.hstack((output[:, :3], np.asarray(q)))
        # un-normalize the predicted and target translations
        output[:, :3] = output[:, :3] * self.max_value

        return output