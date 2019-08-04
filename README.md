# A Convolutional Neural Network for Real Time robot pose estimation by RGB Image

In this project we use slam(gmapping) to collect training dataset(image & robot pose), then using the convolutional neural network(Posenet & Mapnet) to regress the robot pose only by RGB image. (Without depth map)

## Installation Dependencies:

Python2.7

Pytorch

ROS Kinetic

visdom

http://wiki.ros.org/kinetic/Installation/Ubuntu


## How to Run?
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


![image](https://github.com/m5823779/PoseEstimation/blob/master/demo/success.gif)
Demo video (Speed up 5 times )

Or run rosbag

```
$rosbag play -d 3 rosbag
```

## Reference:

https://github.com/NVlabs/geomapnet
