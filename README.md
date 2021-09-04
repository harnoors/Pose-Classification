# Pose-Classification
This project is inspired by DJI's Gesture Piloting Project. This uses Nvidia's [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose) project for real-time pose estimation and using those key-points a Sklearn Classifiers has been train to predict between various classes.
<p align="center"><img src="http://yuml.me/diagram/nofunky/class/[Image/Video{bg:white}]-->[Pose Estimator {bg:cornsilk}]-->[Data-Points{bg:white}]-->[Classfier{bg:cornsilk}]-->[Result{bg:red}]" ></p>

https://user-images.githubusercontent.com/55160417/132078067-1505f137-db2d-4b62-8c12-f02569675434.mp4

In total there are 4 classes
LHR -> 0
RHR -> 1
BHR -> 2
BHU -> 3
