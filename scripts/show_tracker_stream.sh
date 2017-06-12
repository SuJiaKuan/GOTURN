#!/bin/bash

# Choose which GPU the tracker runs on
GPU_ID=0

DEPLOY=nets/tracker.prototxt
CAFFE_MODEL=nets/models/pretrained_model/tracker.caffemodel

build/show_tracker_stream $DEPLOY $CAFFE_MODEL $GPU_ID
