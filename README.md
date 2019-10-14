# Overview
This project seeks to provide a way to understand how spaces are used, with an initial focus on makerspaces
![An example output](data/room_utilization.png?raw=true "Title")
![Example pose detections](data/visualizations/downsample_pose_detection.gif)

# Approach
Recent research on person detection and pose estimation allows us to get robust measurments of where people are in video captured with low-cost cameras. From this, we should be able to generate useful reports on which areas are busiest and when, how do groups interact, and other similar metrics. 
# Quickstart
Run the ./example.sh script
# Install
To install openpose on Jetson:
1. Flash with jetpack 4.2.2
2. Ensure Jetson is in max power mode `sudo nvpmodel -m 0`.
3. Install openpose dependencies following their tutorial here: `https://github.com/CMU-Perceptual-Computing-Lab/openpose`.
4. Checkout this openpose commit `git reset --hard 06d4ea6` (this is to avoid this error: https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1392)
5. Upgrade cmake to 1.15.4 by building from source.
6. Create build directory, cd into it,  run `cmake -DBUILD_PYTHON:=ON ..`.
7. Run make -j6.
8. Test python installation by following these instructions: `https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/modules/python_module.md`.
