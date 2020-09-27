# A Convolutional Neural Network for Real Time robot pose estimation by RGB Image

In this project we use slam(gmapping) to collect training dataset(image & robot pose), then using the convolutional neural network(Posenet & Mapnet) to regress the robot pose only by RGB image. (Without depth map)

### Introduce

Localization is an important issue for navigation, including, self-driving car and indoor navigation for service robots. SLAM has a good performance in indoor localization. Commonly used sensors are mainly divided into lasers or cameras. The advantage of laser SLAM is its high localization accuracy. However, the lack of image information leads to restrictions on some applications, such as finding objects. Visual SLAM relies on RGB image and depth map. It also has good localization performance. Because having RGB images makes it possible to develop more applications in the future. The disadvantage is that a large number of features extracting and matching, cause a large amount of computation. It is easily influenced by missing features, dynamic light sources, and human disturbance. Therefore, this research will focus on the pose estimation only by RGB image, without features extracting and matching. The robot pose is directly regressing by RGB image to achieve the purpose of indoor navigation.

In recent years, deep learning and convolutional neural network (CNN) have achieved good results in many computer vision studies. It can train the entire neural network end-to-end and learn features from the data. There have some studies shown that it is possible to use deep learning to estimate pose by RGB images, such as PoseNet and MapNet. In this study, we use laser SLAM to collect the data, including RGB images and robot pose which is used as the training pairs required by PoseNet and MapNet. Our target is to regress the robot pose based on the current RGB image. Finally, apply this system on the real robot Turtlebot3 Waffle Pi, and combined it with path planning and speed control system which develope by ourself to achieve the goal of navigation.

### Installation Dependencies:

Python2.7

Pytorch

ROS Kinetic

visdom

http://wiki.ros.org/kinetic/Installation/Ubuntu


### How to Run?
1. Using slam to collect data by yourself

   or Download dataset which creat by ourself: Uploading...

2. Connect the datasets live in the data/deepslam_data directory. Let us call your training dataset directory. Make the link by following command:

```
cd data/deepslam_data
ln -s /"Absolute path of training dataset" Env
```

Extract image and robot pose from rosbag save it as png file and txt file

```
cd ../../rosbag2file
./sync_position_with_image.sh -m mode(train or val) -f from_frame_id  -t to_frame_id  -i image_topic -b rosbag_path  -o output_dir -s sequence_name
```

Calculate robot pose and image mean and std value

```
cd ../scripts
python dataset_mean.py  --dataset dataset_name --scene scene_name
```

Training network
Posenet:
```
python train.py --dataset dataset_name --scene scene_name --config_file configs/posenet.ini --model posenet
```

Reduce version (Reduce the amount of training data by 2 times to train)

```
python train.py --dataset dataset_name --scene scene_name --config_file configs/posenet.ini --model posenet --reduce 2
```

After Posenet training 100epoch, enter following command continue traing neural network by Mapnet

Mapnet:
```
python train.py --dataset dataset_name --scene scene_name --config_file configs/step3_skip10.ini --model mapnet --checkpoint logs/"pre_trained_model_by_posenet"
```

Reduce version (If you choose reduce version to train posenet)
```
python train.py --dataset dataset_name --scene scene_name --config_file configs/step3_skip10.ini --model mapnet --checkpoint logs/dataset_name_scene_name_posenet_posenet/epoch_100.pth.tar --reduce 2
```

Test:

Posenet:

```
python eval.py --dataset dataset_name --scene scene_name --model posenet --weights logs/dataset_name_scene_name_posenet_posenet/epoch_100.pth.tar --config_file configs/posenet.ini --val
```
Mapnet:

```
python eval.py --dataset dataset_name --scene scene_name --model mapnet --weights logs/dataset_name_scene_name_mapnet_step3_skip10/epoch_300.pth.tar --config_file configs/step3_skip10.ini --val
```


![image](https://github.com/m5823779/PoseEstimation/blob/master/demo/result.png)

Real Time on turtlebot3:

```
roscore
python -m visdom.server
python pose_estimator.py --dataset dataset_name --scene scene_name --model mapnet --weights logs/dataset_name_scene_name_mapnet_step3_skip10/epoch_300.pth.tar --config_file configs/step3_skip10.ini --plot
```


![image](https://github.com/m5823779/PoseEstimation/blob/master/demo/final_demo.gif)
Demo video (Speed up 3 times )

Or run rosbag

```
$rosbag play -d 3 rosbag
```

## Reference:

https://github.com/NVlabs/geomapnet
