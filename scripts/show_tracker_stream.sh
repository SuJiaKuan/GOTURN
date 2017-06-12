#!/bin/bash

if [ -z "$1" ]
  then
    echo "No stream device specified!"
    echo "Usage: bash `basename "$0"` stream_device"
    exit
fi

# Choose which GPU the tracker runs on
GPU_ID=0

# Set to 0 to pause after each frame
PAUSE_VAL=1

STREAM_DEVICE=$1
DEPLOY=nets/tracker.prototxt
CAFFE_MODEL=nets/models/pretrained_model/tracker.caffemodel

build/show_tracker_stream $DEPLOY $CAFFE_MODEL $STREAM_DEVICE $GPU_ID $PAUSE_VAL
