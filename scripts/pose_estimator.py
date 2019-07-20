#!/usr/bin/env python
import numpy as np
import os.path as osp
import set_paths
import os, cv2, sys, rospy, tf2_ros, argparse, configparser, torch.cuda, tf, math
import geometry_msgs.msg
from visdom import Visdom
from scipy import stats
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from PIL import Image as PIL_Image
from torchvision import transforms, models
from models.posenet import PoseNet, MapNet
from common.train import load_state_dict, step_feedfwd
from common.pose_utils import qexp


class PoseEstimator:
    def __init__(self, args):
        self.dataset = args.dataset
        self.scene = args.scene
        self.weights = args.weights
        self.model = args.model
        self.config_file = args.config_file

        self.config()
        self.eval_net = self.build_network()
        self.stats, self.max_value = self.load_stats()
        self.load_network()
        self.idx = 0

        if self.model.find('mapnet') >= 0:
            self.tmp_img = []

    def config(self):
        settings = configparser.ConfigParser()
        with open(self.config_file, 'r') as f:
            settings.read_file(f)
        self.seed = settings.getint('training', 'seed')
        section = settings['hyperparameters']
        self.dropout = section.getfloat('dropout')

        if self.model.find('mapnet') >= 0:
            self.skip = section.getint('skip')
            self.steps = section.getint('steps')
        if args.opt:
            self.interval = section.getint('interval')
            self.stride = section.getint('stride')

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
        stats = np.loadtxt(stats_filename)

        # read mean and stdev for un-normalizing predictions
        pose_stats_file = osp.join(data_dir, self.scene, 'pose_stats.txt')
        max_value = np.loadtxt(pose_stats_file)  # max value

        return stats, max_value

    def transform(self, img):
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

        cv2.imshow('Raw Image', cv2.resize(img, (img.shape[1], img.shape[0]), interpolation=PIL_Image.BILINEAR))
        cv2.waitKey(1)

        # Transform image from array to PIL image
        img = PIL_Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = self.transform(img)

        if self.model.find('mapnet') >= 0:
            if len(self.tmp_img) > 2 * self.skip:
                self.tmp_img.remove(self.tmp_img[0])
            self.tmp_img.append(img)

            skips = self.skip * np.ones(self.steps - 1)
            offsets = np.insert(skips, 0, 0).cumsum()
            offsets -= offsets[-1]
            offsets = offsets.astype(np.int)

            if self.idx > 2 * self.skip:
                index = 2 * self.skip + offsets
            else:
                index = self.idx + offsets

            index = np.minimum(np.maximum(index, 0), len(self.tmp_img) - 1)
            clip = [self.tmp_img[i] for i in index]
            img = torch.stack([c for c in clip], dim=0)

        img = img.unsqueeze(0)
        # output : 1 x 6 or 1 x STEPS x 6
        _, pose = step_feedfwd(img, self.eval_net, CUDA, train=False)
        s = pose.size()
        pose = pose.cpu().data.numpy().reshape((-1, s[-1]))

        # normalize the predicted quaternions
        q = [qexp(p[3:]) for p in pose]
        pose = np.hstack((pose[:, :3], np.asarray(q)))

        # un-normalize the predicted and target translations
        pose[:, :3] = pose[:, :3] * self.max_value
        if args.model.find('mapnet') >= 0:
            pred_pose = pose[-1]
        else:
            pred_pose = pose[0]
        self.idx += 1

        return pred_pose

class Poses:
    def __init__(self, args):
        rospy.init_node("pose_estimate_node")
        if args.plot:
            self.vis = Visdom()

        if args.opt:
            self.tmp_pose = np.empty((0, 7))
            self.dis = []

        if args.ground_truth:
            self.from_frame_id = 'map'
            self.to_frame_id = 'base_footprint'
            self.tf_buffer = tf2_ros.Buffer()
            self.listener = tf2_ros.TransformListener(self.tf_buffer)

            self.idx, y = 0, 0
            self.win_angle_error = self.vis.line(np.array([y]), np.array([self.idx]), opts=dict(title='Angel_error', width=800, height=800))

        self.bridge = CvBridge()
        self.pose_estimator = PoseEstimator(args)
        rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)
        rospy.spin()

    def callback(self, img):
        if args.ground_truth:
            t = self.tf_buffer.lookup_transform(self.from_frame_id, self.to_frame_id, rospy.Time())
            tran = t.transform.translation
            rot = t.transform.rotation
            slam_pose = [tran.x, tran.y, tran.z, rot.w, rot.x, rot.y, rot.z]
        else:
            slam_pose = None

        cv_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        pred_pose = self.pose_estimator.estimation(cv_image)

        if args.opt:
            if len(self.tmp_pose) == self.pose_estimator.interval:
                self.tmp_pose[0][(np.abs(stats.zscore(self.tmp_pose, axis=0)) < 1).all(axis=0)]
                mean_value = np.mean(self.tmp_pose[:, [0, 1, 2]], axis=0)
                for k in range(len(self.tmp_pose)):
                    self.dis.append(np.sqrt((self.tmp_pose[k][0] - mean_value[0]) ** 2 + (self.tmp_pose[k][1] - mean_value[1]) ** 2 + (self.tmp_pose[k][2] - mean_value[2]) ** 2))
                min_index = self.dis.index(np.min(self.dis))
                pred_pose = self.tmp_pose[min_index]

                self.publisher(pred_pose)
                print 'x = {:.2f} | y = {:.2f} | z = {:.2f} | w = {:.2f} | p = {:.2f} | q = {:.2f} | r = {:.2f}' \
                    .format(pred_pose[0], pred_pose[1], pred_pose[2], pred_pose[3], pred_pose[4], pred_pose[5], pred_pose[6])
                if args.plot:
                    self.plot(pred_pose, slam_pose)
                self.dis = []

            if len(self.tmp_pose) > self.pose_estimator.interval:
                for i in range(self.pose_estimator.stride):
                    self.tmp_pose = np.delete(self.tmp_pose, 0, axis=0)
            self.tmp_pose = np.vstack((self.tmp_pose, pred_pose))
        else:
            self.publisher(pred_pose)
            print'x = {:.2f} | y = {:.2f} | z = {:.2f} | w = {:.2f} | p = {:.2f} | q = {:.2f} | r = {:.2f}' \
                .format(pred_pose[0], pred_pose[1], pred_pose[2], pred_pose[3], pred_pose[4], pred_pose[5], pred_pose[6])
            if args.plot:
                self.plot(pred_pose, slam_pose)

    def publisher(self, pred_pose):
        br = tf2_ros.TransformBroadcaster()
        t = geometry_msgs.msg.TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = 'predict_pose'
        t.transform.translation.x = pred_pose[0]
        t.transform.translation.y = pred_pose[1]
        t.transform.translation.z = pred_pose[2]
        t.transform.rotation.x = pred_pose[4]
        t.transform.rotation.y = pred_pose[5]
        t.transform.rotation.z = pred_pose[6]
        t.transform.rotation.w = pred_pose[3]

        br.sendTransform(t)

    def plot(self, pred_pose, slam_pose):
        if slam_pose is not None:
            plot_pose = np.array([[pred_pose[0], pred_pose[1]], [slam_pose[0], slam_pose[1]]])
            marker_color = np.array([[255, 0, 0], [0, 255, 0]])

            pred_quaternion = (pred_pose[4], pred_pose[5], pred_pose[6], pred_pose[3])
            slam_quaternion = (slam_pose[4], slam_pose[5], slam_pose[6], slam_pose[3])
            pred_euler = tf.transformations.euler_from_quaternion(pred_quaternion)
            slam_euler = tf.transformations.euler_from_quaternion(slam_quaternion)
            pred_yaw = math.degrees(pred_euler[2])
            slam_yaw = math.degrees(slam_euler[2])
            diff_angle = abs(pred_yaw - slam_yaw)
            self.idx += 1
            self.vis.line(np.array([diff_angle]), np.array([self.idx]), self.win_angle_error, update="append")
            try:
                self.win_real_time = self.vis.scatter(plot_pose, win=self.win_real_time, update='new', opts=dict(markersize=5, markercolor=marker_color))
            except:
                self.win_real_time = self.vis.scatter(plot_pose, opts=dict(title='Trajectory', width=800, height=800, xtickmin=-10, xtickmax=10, ytickmin=-10, ytickmax=10, markersize=5, markercolor=marker_color))

        else:
            plot_pose = np.array([[pred_pose[0], pred_pose[1]]])
            marker_color = np.array([[255, 0, 0]])

        # Trajectory
        try:
            self.win_trajectory = self.vis.scatter(plot_pose, win=self.win_trajectory, update='append', opts=dict(markersize=2, markercolor=marker_color))
        except:
            self.win_trajectory = self.vis.scatter(plot_pose, opts=dict(title='Trajectory', width=800, height=800, xtickmin=-10, xtickmax=10, ytickmin=-10, ytickmax=10, markersize=2, markercolor=marker_color))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real time script for PoseNet and MapNet')
    parser.add_argument('--dataset', type=str, default='Env', help='Dataset')
    parser.add_argument('--scene', type=str, default='Lab', help='Scene name')
    parser.add_argument('--weights', type=str, default='logs/Env_Lab_mapnet_mapnet_learn_beta_learn_gamma/epoch_300.pth.tar', help='trained weights to load')
    parser.add_argument('--model', type=str, default='mapnet', help='Model to use')
    parser.add_argument('--config_file', type=str, default='configs/mapnet.ini', help='configuration file')
    parser.add_argument('--plot', action='store_true', help='Plot on visdom')
    parser.add_argument('--ground_truth', action='store_true', help='With Ground Truth from SLAM')
    parser.add_argument('--opt', action='store_true', help='Optimize')
    args = parser.parse_args()
    estimatior = Poses(args)