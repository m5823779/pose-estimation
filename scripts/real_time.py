import cv2
import torch
import argparse
import configparser
import numpy as np

from scipy import stats
from PIL import Image
from visdom import Visdom
from estimator import Estimator

def main():
    vis = Visdom()

    settings = configparser.ConfigParser()
    with open(args.config_file, 'r') as f:
        settings.read_file(f)
    section = settings['hyperparameters']
    if args.model.find('mapnet') >= 0:
        skip = section.getint('skip')
        steps = section.getint('steps')
    if args.opt:
        interval = section.getint('interval')
        stride = section.getint('stride')

    agent = Estimator(dataset=args.dataset, scene=args.scene, weights=args.weights, model=args.model, config_file=args.config_file)

    idx = 0
    tmp_img = []
    tmp_pose = np.empty((0, 7))
    dis = []
    while True:
        # img = cv2.imread('/home/airlab/Relocalization_data/Env/Lab/seq-04/frame-{:06d}.color.png'.format(idx))
        img = cv2.imread('/home/airlab/Relocalization_data/Env/Lab/seq-04/frame-{:06d}.color.png'.format(idx))

        cv2.imshow('Raw_Image', cv2.resize(img, (img.shape[1] / 2, img.shape[0] / 2), interpolation=Image.BILINEAR))
        cv2.waitKey(1)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = agent.transform(img)

        if args.model.find('mapnet') >= 0:
            if len(tmp_img) > 2 * skip:
                tmp_img.remove(tmp_img[0])
            tmp_img.append(img)

            skips = skip * np.ones(steps - 1)  # [10.  10.]
            offsets = np.insert(skips, 0, 0).cumsum()  # [0.  10.  20.]
            offsets -= offsets[len(offsets) - 1]  # [-20.  -10.  0.]
            offsets = offsets.astype(np.int)  # [-20  -10  0]

            if idx > 2 * skip:
                index = 2 * skip + offsets
            else:
                index = idx + offsets

            index = np.minimum(np.maximum(index, 0), len(tmp_img) - 1)
            clip = [tmp_img[i] for i in index]
            img = torch.stack([c for c in clip], dim=0)

        img = img.unsqueeze(0)
        pose = agent.estimation(img)

        if args.model.find('mapnet') >= 0:
            pose = pose[-1]
        else:
            pose = pose[0]

        if args.opt:
            if len(tmp_pose) == interval:
                tmp_pose[0][(np.abs(stats.zscore(tmp_pose, axis=0)) < 1).all(axis=0)]
                mean_value = np.mean(tmp_pose[:, [0, 1, 2]], axis=0)
                for k in range(len(tmp_pose)):
                    dis.append(np.sqrt((tmp_pose[k][0] - mean_value[0]) ** 2 + (tmp_pose[k][1] - mean_value[1]) ** 2 + (tmp_pose[k][2] - mean_value[2]) ** 2))
                min_index = dis.index(np.min(dis))
                pose = tmp_pose[min_index]
                print 'Pose: x = {:.3f} | y = {:.3f} | z = {:.3f}'.format(pose[0], pose[1], pose[2])
                plot_pose = np.array([[pose[0], pose[1]]])
                try:
                    win = vis.scatter(plot_pose, win=win, update='append')
                except:
                    win = vis.scatter(plot_pose, opts=dict(title='Trajectory', width=800, height=800, xtickmin=-10, xtickmax=10, ytickmin=-10, ytickmax=10, markersize=3, markercolor=np.array([[255, 0, 0]])))
                dis = []
            if len(tmp_pose) > interval:
                for _ in range(stride):
                    tmp_pose = np.delete(tmp_pose, 0, axis=0)
            tmp_pose = np.vstack((tmp_pose, pose))

        else:
            print 'Pose: x = {:.3f} | y = {:.3f} | z = {:.3f}'.format(pose[0], pose[1], pose[2])

            plot_pose = np.array([[pose[0], pose[1]]])
            try:
                win = vis.scatter(plot_pose, win=win, update='append')
            except:
                win = vis.scatter(plot_pose, opts=dict(title='Trajectory', width=800, height=800, xtickmin=-10, xtickmax=10, ytickmin=-10, ytickmax=10, markersize=3, markercolor=np.array([[255,0,0]])))
        idx += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real time script for PoseNet and MapNet')
    parser.add_argument('--dataset', type=str, default='Env', help='Dataset')
    parser.add_argument('--scene', type=str, default='Lab', help='Scene name')
    parser.add_argument('--weights', type=str, help='trained weights to load')
    parser.add_argument('--model', type=str, choices=('posenet', 'mapnet'), help='Model to use')
    parser.add_argument('--config_file', type=str, help='configuration file')
    parser.add_argument('--opt', action='store_true', help='Optimize')
    args = parser.parse_args()
    main()



