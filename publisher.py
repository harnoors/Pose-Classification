import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time, sys
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import os.path
import numpy as np
import pandas as pd
import pickle
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

debug = False

# import trt_pose model and optimize it
'''
hnum: 0 based human index
kpoint : keypoints (float type range : 0.0 ~ 1.0 ==> later multiply by image width, height
'''
with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])


print('------ model = resnet--------')
MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
WIDTH = 224
HEIGHT = 224

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

if os.path.exists(OPTIMIZED_MODEL) == False:
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

# ROS2 publisher to publish the classifcation results
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


# extracts data points from human[0]
def get_keypoint(humans, hnum, peaks):
    #check invalid human index
    kpoint = []
    klist = []
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
            peak1 = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak1)
            klist.append(float(peak[0]))
            klist.append(float(peak[1]))
        else:    
            peak = (j, None, None)
            kpoint.append(peak)
            klist.append(0)
            klist.append(0)
    return kpoint, klist

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

# import Sklearn Model
clf = pickle.load(open("models/RandomForestClf.pkl", "rb"))

# image -> extract datapoints using trt_pose -> classifies using sklearn classifier
def execute(img, src, t):
    # 1. process image
    color = (0, 255, 0)
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    fps = 1.0 / (time.time() - t)
    
    # 2. get keypoints for human[0]
    keypoints, kplist = get_keypoint(objects, 0, peaks)
    
    # 3. transform list to dataframe
    df = pd.DataFrame(kplist).T
    
    # 4. predict results
    results = clf.predict(df)
    resultsP = clf.predict_proba(df)
    print("the predict is: :", results,"  with prob:", resultsP[0][results])
    
    # 5. publish predicted results
    minimal_publisher.send_cl([float(results), float(resultsP[0][results])])
    
    if resultsP[0][results] > 0.3:
        cv2.putText(src, "pos: " + str(results), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        
    for j in range(len(keypoints)):
        if keypoints[j][1]:
            x = round(keypoints[j][2] * WIDTH * X_compress)
            y = round(keypoints[j][1] * HEIGHT * Y_compress)
            cv2.circle(src, (x, y), 3, color, 2)
            cv2.putText(src , "%d" % int(keypoints[j][0]), (x + 5, y),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
            cv2.circle(src, (x, y), 3, color, 2)
   
    print("FPS:%f "%(fps))
    cv2.putText(src , "FPS: %f" % (fps), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    out_video.write(src)  # writing to output stream 
    cv2.imshow('key', src)
    cv2.waitKey(1)

rclpy.init(args=None)
minimal_publisher = MinimalPublisher()

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)
X_compress = 1280.0 / WIDTH * 1.0
Y_compress = 720.0 / HEIGHT * 1.0

# CSI camera setup
cap_str = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)NV12, framerate=(fraction)24/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

cap = cv2.VideoCapture(cap_str)
ret_val, img = cap.read()
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

# output video
out_video = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (1280, 720))

count = 0

if cap is None:
    print("Camera Open Error")
    sys.exit(0)

while cap.isOpened() and count < 200:
    t = time.time()
    ret_val, dst = cap.read()
    if ret_val == False:
        print("Camera read Error")
        break

    img = cv2.resize(dst, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    execute(img, dst, t)
    count += 1

cv2.destroyAllWindows()
out_video.release()
cap.release()
