#!/bin/bash
PROGRAMNAME=$0

DEFAULT_FROM_FRAME_ID=map
DEFAULT_TO_FRAME_ID=base_footprint
DEFAULT_IMAGE_TOPIC=/kinect2/qhd/image_color_rect

ROSBAG_RATE=1

function error_exit
{
  echo "${PROGRAMNAME}: ${1:-"Unknown Error"}" 1>&2
  exit 1
}

function usage {
    echo "usage: ${PROGRAMNAME} [-m MODE] [-f FROM] [-t TO] [-i TOPIC] [-b ROSBAG] [-o OUTPUT] [-s SEQ]"
    echo
    echo "Extracts TF transformations FROM a given frame TO a given frame and"
    echo "synchronizes them with images with from a given image TOPIC."
    echo
    echo "   [-m MODE]      MODE of dataset"
    echo "   [-f FROM]      FROM frame id"
    echo "   [-t TO]        TO frame id"
    echo "   [-i TOPIC]     Image topic name"
    echo "   [-b ROSBAG]    Path to a ROSBAG file containing TF and image topics"
    echo "   [-o OUTPUT]    Path to an output directory"
    echo "   [-s SEQ]       Sequence name"
    echo "   [-h]           Prints this help"
    echo
    exit 1
}

from_frame_id=${DEFAULT_FROM_FRAME_ID}
to_frame_id=${DEFAULT_TO_FRAME_ID}
image_topic=${DEFAULT_IMAGE_TOPIC}

while getopts 'm:f:t:i:b:o:s:h' flag; do
  case "${flag}" in
    m) data_mode="${OPTARG}" ;;
    f) from_frame_id="${OPTARG}" ;;
    t) to_frame_id="${OPTARG}" ;;
    i) image_topic="${OPTARG}" ;;
    b) rosbag="${OPTARG}" ;;
    o) output_dir="${OPTARG}" ;;
    s) sequence_name="${OPTARG}" ;;
    h) usage ;;
    *) error_exit "Unexpected option ${flag}" ;;
  esac
done

if [[ -z $rosbag ]]; then error_exit "Argument [-b ROSBAG] is mandatory"; fi
if [[ -z $output_dir ]]; then error_exit "Argument [-o OUTPUT] is mandatory"; fi

echo "Extracting transform from ${from_frame_id} to ${to_frame_id}"
echo "Syncing the transform with ${image_topic}"

all_transforms_dir=${output_dir}/${sequence_name}/numpy
mkdir -p ${all_transforms_dir}

echo "Saving all extracted transformations in numpy file"
numpy_file=${all_transforms_dir}/poses.npy
python tf_extractor.py -f ${from_frame_id} -t ${to_frame_id} -o ${numpy_file} -v &

echo 'TF extraction node started!'
rosbag play -d 3 --clock -r ${ROSBAG_RATE} ${rosbag}

echo 'Finished extracting TF transforms'
kill -INT %1

echo 'Starting image extract and sync'
python extract_and_sync.py -m ${data_mode} -b ${rosbag} -a ${numpy_file} -t ${image_topic} ${output_dir}/${sequence_name}

sleep 10
echo 'Done'

