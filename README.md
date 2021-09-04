# Pose-Classification
This project is inspired by DJI's Gesture Piloting Project. This uses Nvidia's [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose) project for real-time pose estimation and using those key-points a Sklearn Classifiers has been train to predict between various classes.
<p align="center"><img src="http://yuml.me/diagram/nofunky/class/[Image/Video{bg:white}]-->[Pose Estimator {bg:cornsilk}]-->[Data-Points{bg:white}]-->[Classfier{bg:cornsilk}]-->[Result{bg:red}]" ></p>

# Demo
<p align="center">
  <img src="images/pose.gif" alt="animated" />
</p>

## Introduction
In total there are **4 classes**: 

<p align="center">
  1. Left Hand Raised (LHR) -> 0 <br>
  <img src="https://github.com/harnoors/Pose-Classification/blob/main/images/LHR/IMG_7519.JPEG" alt="animated" width="432" height="223">
</p>
<br>
<p align="center">
  2. Right Hand Raised (RHR) -> 1 <br>
  <img src="https://github.com/harnoors/Pose-Classification/blob/main/images/RHR/IMG_7530.JPEG" alt="animated" width="432" height="223">
</p>
<br>
<p align="center">
  3. Both Hands Raised (BHR) -> 2 <br>
  <img src="https://github.com/harnoors/Pose-Classification/blob/main/images/BHR/IMG_7579.JPEG" alt="animated" width="432" height="223">
</p>
<br>
<p align="center">
  4. Both Hands up (BHU) -> 3 <br>
  <img src="https://github.com/harnoors/Pose-Classification/blob/main/images/BHU/IMG_7599.JPEG" alt="animated" width="432" height="223">
</p>

