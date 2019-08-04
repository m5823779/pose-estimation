#!/usr/bin/env python
import numpy as np
import os.path as osp
import set_paths
import os, cv2, sys, rospy, tf2_ros, argparse, configparser, torch.cuda, tf, math
import geometry_msgs.msg
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
        self.stats, self.pose_m, self.pose_s = self.load_stats()
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
        pose_m, pose_s = np.loadtxt(pose_stats_file)  # max value

        return stats, pose_m, pose_s

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

        # cv2.imshow('Raw Image', cv2.resize(img, (img.shape[1], img.shape[0]), interpolation=PIL_Image.BILINEAR))
        # cv2.waitKey(1)

        # Transform image from array to PIL image
        img = PIL_Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = self.transform(img)

        if self.model.find('mapnet') >= 0:
            if len(self.tmp_img) > (self.steps - 1) * self.skip:
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
        pose[:, :3] = (pose[:, :3] * self.pose_s) + self.pose_m

        if args.model.find('mapnet') >= 0:
            pred_pose = pose[-1]
        else:
            pred_pose = pose[0]

        self.idx += 1

        return pred_pose


class Poses:
    def __init__(self, args):
        rospy.init_node("pose_estimate_node")
        self.bridge = CvBridge()
        self.pose_estimator = PoseEstimator(args)
        if args.opt:
            self.tmp_pose = np.zeros((1, 7))
            self.reset_para()
        rospy.Subscriber("/usb_cam/image_raw", Image, self.callback, buff_size=640*480*30, queue_size=1)
        rospy.spin()

    def callback(self, img):
        cv_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        pred_pose = self.pose_estimator.estimation(cv_image)

        if args.opt:
            self.noise_filter(pred_pose)
        else:
            self.publisher(pred_pose)
            print'x = {:.2f} | y = {:.2f} | z = {:.2f} | w = {:.2f} | p = {:.2f} | q = {:.2f} | r = {:.2f}' \
                .format(pred_pose[0], pred_pose[1], pred_pose[2], pred_pose[3], pred_pose[4], pred_pose[5], pred_pose[6])

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

    def noise_filter(self, pred_pose):
        if self.tmp_pose.any() == 0:
            self.tmp_pose[0, :] = np.array([pred_pose[0], pred_pose[1], pred_pose[2], pred_pose[3], pred_pose[4], pred_pose[5], pred_pose[6]])

        delta_x = pred_pose[0] - self.tmp_pose[0, 0]
        delta_y = pred_pose[1] - self.tmp_pose[0, 1]
        move_dis = np.sqrt(delta_x ** 2 + delta_y ** 2)

        current_quaternion = (pred_pose[4], pred_pose[5], pred_pose[6], pred_pose[3])
        past_quaternion = (self.tmp_pose[0, 4], self.tmp_pose[0, 5], self.tmp_pose[0, 6], self.tmp_pose[0, 3])
        current_euler = tf.transformations.euler_from_quaternion(current_quaternion)
        past_euler = tf.transformations.euler_from_quaternion(past_quaternion)
        move_angle = abs(current_euler[2] - past_euler[2])

        if move_angle > math.pi:
            move_angle = 2 * math.pi - move_angle

        if self.unusual == 40:
            self.tmp_pose[0, :] = np.array([pred_pose[0], pred_pose[1], pred_pose[2], pred_pose[3], pred_pose[4], pred_pose[5], pred_pose[6]])
            self.reset_para()

        if move_dis > self.unusual_displacement_threshold or move_angle > self.unusual_angle_threshold:
            self.unusual += 1
            self.unusual_displacement_threshold += 0.02
            self.unusual_angle_threshold += 0.05
            if self.unusual % 5 ==0:
                print 'Continuous unusual {:02d} times: '.format(self.unusual)
            pred_pose = self.tmp_pose[0]

        else:
            self.tmp_pose[0, :] = np.array([pred_pose[0], pred_pose[1], pred_pose[2], pred_pose[3], pred_pose[4], pred_pose[5], pred_pose[6]])
            self.reset_para()

        self.publisher(pred_pose)
        print'x = {:.2f} | y = {:.2f} | z = {:.2f} | w = {:.2f} | p = {:.2f} | q = {:.2f} | r = {:.2f}' \
            .format(pred_pose[0], pred_pose[1], pred_pose[2], pred_pose[3], pred_pose[4], pred_pose[5], pred_pose[6])

    def reset_para(self):
        self.unusual = 0
        self.unusual_displacement_threshold = 0.02
        self.unusual_angle_threshold = 0.1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real time script for PoseNet and MapNet')
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--scene', type=str, help='Scene name')
    parser.add_argument('--weights', type=str, help='trained weights to load')
    parser.add_argument('--model', type=str,  help='Model to use')
    parser.add_argument('--config_file', type=str, help='configuration file')
    parser.add_argument('--opt', action='store_true', help='Optimize')
    args = parser.parse_args()
    estimatior = Poses(args)
