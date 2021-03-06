# Pose-Classification
This project is inspired by DJI's Gesture Piloting Project. This uses Nvidia's [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose) project for real-time pose estimation and using those key-points a Sklearn Classifiers has been train to predict between various classes.
<p align="center"><img src="http://yuml.me/diagram/nofunky/class/[Image/Video{bg:white}]-->[Pose Estimator {bg:cornsilk}]-->[Data-Points{bg:white}]-->[Classfier{bg:cornsilk}]-->[Result{bg:red}]" ></p>

**Note: This project was implemented and tested on Jetson Nano.**

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

# Demo
<p align="center">
  <img src="images/pose.gif" alt="animated" />
</p>

# Getting Started

### 1. install Dependencies 


* Install PyTorch and Torchvision. To do this on NVIDIA Jetson, Its recommended following [this](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-9-0-now-available/72048)

* Install torch2trt
  ```
  git clone https://github.com/NVIDIA-AI-IOT/torch2trt
  cd torch2trt
  sudo python3 setup.py install --plugins

  ```
* Install other miscellaneous packages
  ```
  sudo pip3 install tqdm cython pycocotools
  sudo apt-get install python3-matplotlib
  ```
* Install trt_pose
  ```
  git clone https://github.com/NVIDIA-AI-IOT/trt_pose
  cd trt_pose
  sudo python3 setup.py install
  ```
* Jupyter
  - Jupyter Notebook (IF THIS FAILS THEN FOLLOW POINT 3)
      ```
      sudo apt install nodejs npm
      sudo apt install python3-pip
      sudo pip3 install jupyter jupyterlab
      sudo jupyter labextension install @jupyter-widgets/jupyterlab-manager
      jupyter lab --generate-config
      ```
  - Jupyter if npm fails (failed for me in Jetpack 4.5.1)
      ```
      # install node manually
      wget https://nodejs.org/dist/v16.0.0/node-v16.0.0-linux-arm64.tar.gz
      tar -zxvf node-v16.0.0-linux-arm64.tar.gz
      sudo cp node-v16.0.0-linux-arm64 /etc/node-v16.0.0 -r
      sudo ln -s /etc/node-v16.0.0/bin/node /usr/bin/node
      sudo ln -s /etc/node-v16.0.0/bin/npm /usr/bin/npm
      sudo ln -s /etc/node-v16.0.0/bin/npx /usr/bin/npx
      #follow the rest of steps
      sudo apt install python3-pip
      sudo pip3 install jupyter jupyterlab
      sudo jupyter labextension install @jupyter-widgets/jupyterlab-manager
      jupyter lab --generate-config
      ```
 
* Scipy (1.1.0)
  ```
  pip install scipy
  ```
* Sklearn (0.0)
  ```
  pip install sklearn
  ```
* Scikit-Learn (0.24.dev0)
  ```
  pip install scikit-learn=0.24.dev0
  ```
* Building OpenCV-3.4.6 with GStreamer ON
    - You can check if Gstremer is ON using
        ```
        import cv2
        print(cv2.getBuildInformation())
        ```
    - Run this Script here.
    - https://github.com/jkjung-avt/jetson_nano/blob/master/install_opencv-3.4.6.sh
 
* ROS2 for ubuntu 18.04 on Jetson Nano:
  - I had to do ```pip install lark``` before running the above script.
  - Run the script here: https://github.com/jetsonhacks/installROS2
  - After runnning the script ROS2 got installed in not so usual loaction, for mw the directory was ```/opt/ros/foxy/install/setup.bash```
* Pytorch + trt_pose
  Follow the tutorials here
  PyTorch : https://spyjetson.blogspot.com/2019/10/jetsonnano-human-pose-estimation-using_16.html   
  trt_pose: https://spyjetson.blogspot.com/2019/12/jetsonnano-human-pose-estimation-using.html
  
  This is an extract from above tutorials:

    1. Delete old versions of PyTorch (IF ANY)
        ```
        First check pre-installed PyTorch. If there is no PyTorch version already
        installed, proceed to the next step.
        root@spypiggy-nano:/usr/local/src/detr# pip3 freeze|grep torch
        torch==1.1.0
        torchvision==0.3.0
        root@spypiggy-nano:/usr/local/src/detr# pip3 uninstall torchvision==0.3.0
        root@spypiggy-nano:/usr/local/src/detr# pip3 uninstall torch==1.1.0
        ```
    2. Download PyTorch for Jetson Nano
        ```
        sudo apt-get update
        wget https://nvidia.box.com/shared/static/yr6sjswn25z7oankw8zy1roow9cy5ur1.whl
        -O torch-1.6.0-cp36-cp36m-linux_aarch64.whl
        sudo apt-get install python3-pip libopenblas-base libopenmpi-dev
        pip3 install Cython
        pip3 install torch-1.6.0-cp36-cp36m-linux_aarch64.whl
        ```
    3. Download Torchvision for Jetson Nano
        ```
        sudo apt-get install libjpeg-dev zlib1g-dev
        wget https://github.com/pytorch/vision/archive/v0.6.1.tar.gz
        tar -xvzf v0.6.1.tar.gz
        cd v0.6.1
        sudo python3 setup.py install
        apt-get install libfreetype6-dev
        pip3 uninstall pillow
        pip3 install --no-cache-dir pillow
        ```
    4. Torch2trt:
        ```
        cd /usr/local/src
        git clone https://github.com/NVIDIA-AI-IOT/torch2trt
        cd torch2trt
        python3 setup.py install
        Trt_pose:
        pip3 install tqdm cython pycocotools
        apt-get install python3-matplotlib
        cd /usr/local/src
        git clone https://github.com/NVIDIA-AI-IOT/trt_pose
        cd trt_pose
        python3 setup.py install
        ```  
# Explaining All the files
* **CSI.py**: This python file takes in feed from the CSI camera, displays and saves the results as output.mp4
* **detect_video.py**: This python file takes in a pre-recorder video, displays and saves the results as output.mp4
  - The source of the video can be updated in line 166 ```video = 'source_of_your_video' ``` 
* **ROS2 setup**:
  -  publisher.py
      -  This python file takes in feed from the CSI camera, displays, saves (results as output.mp4), and publishes the results at ```pose_classification``` channel in  Float32Array.
          - The results are publised an array in a format: ```[predicted_class, probability_of_that_prediction]```
          - If you wish to change how (format) or add in what interval the results are published, that can be done near line 137 of  ```publisher.py```  
            - ```minimal_publisher.send_cl([float(results), float(resultsP[0][results])])``` 
      -  This piece of code is a snip from ```publisher.py``` and is used to publish classification results.
          - ``` 
            class MinimalPublisher(Node):
              def __init__(self):
                  super().__init__('sphero_node')
                  self.publish_cl = self.create_publisher(Float32MultiArray, 'pose_classification', 10)

              # send_cl() will take a data and send it to pose_classification channel
              def send_cl(self, data):
                  msg = Float32MultiArray()
                  msg.data = data
                  self.publish_cl.publish(msg)
                  if debug: self.get_logger().info('Publishing: "%s"' % msg.data)
              ```  
    
  -  subscriber.py
      -  This python file subscribes to the ```pose_classification``` channel and simply prints out the results. 
          - ```print("the predicted class is: ", int(result),"  with probability:", resultPr)``` 
      -  This piece of code is a snip from ```subscriber.py```
          -  ```
             class MinimalSubscriber(Node):
                def __init__(self):
                    super().__init__('minimal_subscriber')
                    self.subscription = self.create_subscription(Float32MultiArray,'pose_classification',self.cl_callback, 10)
                    self.subscription

                async def cl_callback(self, msg):
                    msg = msg.data
                    result = int(msg[0])
                    resultPr = float(msg[1])
                    ## Use result and probability to make decisions 
                    print("the predicted class is: ", int(result),"  with probability:", resultPr)
             ```
           - The cl_callback() function can be updated to make decesions or possibly publish some commands to drone/rover. 
     
# Running the project
1. Clone the Repo: 
    ```
    git clone https://github.com/harnoors/Pose-Classification.git
    cd Pose-Classification
    ```
2. Running the ROS2 Node:
    - ```python3 publisher.py```
         
          Note: This will take a while to start the first time because the
                pose_detection trt model gets optimized during the first run. 
    - After the camera preview starts, open another terminal to see what data is being published ```ros2 topic list```
    - ```pyhton3 subscriber.py```
      - This could be used to print the classification results coming in from the publisher. Additionaly this could also be updated to publish commands to drone. 
       
3. CSI camera:
    - ```python3 CSI.py```
   
4. Video (pre-recorded):
    - ```python3 detect_video.py```
    - NOTE: change the directory of the video in line 166 ```video = 'source_of_your_video' ``` 
   


# Pipeline
1. TODO: 
THIS PART WILL EXPLIAN HOW TO COLLECT DATA TO TRAIN YOUR OWN MODEL AND ADD ANOTHER CLASSES IF YOU WISH TO.

