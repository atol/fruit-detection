"""
Based on: https://blog.paperspace.com/tag/series-yolo/
"""

from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pickle as pkl
import pandas as pd
import cv2
import argparse
import time
import random
import os 
import os.path as osp

from utils import *
from darknet import Darknet


def arg_parse():
    """
    Parse arguments to the detection module
    """
    parser = argparse.ArgumentParser(description="YOLOv3 Detection Module")
   
    parser.add_argument("--images", dest="images", help="Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--output", dest="output", help="Directory to store detections to",
                        default="output", type=str)
    parser.add_argument("--batch", dest="batch", help="Batch size", default=1)
    parser.add_argument("--conf_thres", dest="conf_thres", help="Object confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thres", dest="nms_thres", help="NMS threshhold", default=0.4)
    parser.add_argument("--cfg", dest="cfgfile", help="Config file", default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest="weights", help="Weights file", default="weights/yolov3.weights", type=str)
    parser.add_argument("--img_size", dest="img_size", 
                        help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    
    return parser.parse_args()
    
args = arg_parse()
images = args.images
batch_size = int(args.batch)
conf_thres = float(args.conf_thres)
nms_thres = float(args.nms_thres)
cuda = torch.cuda.is_available()
start = 0

num_classes = 80    # For COCO
classes = load_classes("data/coco.names")

# Use GPU if available
device = torch.device("cuda" if cuda else "cpu")

# Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile).to(device)
model.load_weights(args.weights)
print("Network successfully loaded")

model.net_info["height"] = args.img_size
input_dim = int(model.net_info["height"])
assert input_dim % 32 == 0
assert input_dim > 32

# Set model in evaluation mode
model.eval()

# Read input images
read_dir = time.time() # Checkpoint time

try:
    img_list = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    img_list = []
    img_list.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print ("No file or directory with the name {}".format(images))
    exit()

# If output directory does not exist, create it
if not os.path.exists(args.output):
    os.makedirs(args.output)

# Load images
load_batch = time.time() # Checkpoint time
loaded_imgs = [cv2.imread(x) for x in img_list]

img_batches = list(map(preprocess, loaded_imgs, [input_dim for x in range(len(img_list))]))

# List containing dimensions of original images
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

input_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_imgs]
input_dim_list = FloatTensor(input_dim_list).repeat(1,2)

# Create the batches
leftover = 0

if (len(input_dim_list) % batch_size):
   leftover = 1

if batch_size != 1:
   num_batches = len(img_list) // batch_size + leftover
   img_batches = [torch.cat((img_batches[i*batch_size : min((i+1)*batch_size,
                       len(img_batches))]))  for i in range(num_batches)]

# Detection loop
output = torch.Tensor()
detect_loop = time.time()

for i, batch in enumerate(img_batches):
    # Load the image
    start = time.time()

    if cuda:
        batch = batch.cuda()

    with torch.no_grad():
        prediction = model(batch, cuda)
        prediction = non_max_suppression(prediction, conf_thres, num_classes, nms_thres=nms_thres)

    end = time.time()

    if prediction.size(0) == 0: # If prediction is empty, there is no detection so skip rest of loop
        for img_num, img in enumerate(img_list[i*batch_size: min((i+1)*batch_size, len(img_list))]):
            img_id = i*batch_size + img_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(img.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:,0] += i*batch_size    # Transform the attribute from index in batch to index in img_list 

    # Concatenate prediction tensors of all images that we have performed detections on
    output = torch.cat((output, prediction))

    for img_num, image in enumerate(img_list[i*batch_size: min((i +  1)*batch_size, len(img_list))]):
        img_id = i*batch_size + img_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == img_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if cuda:
        torch.cuda.synchronize()

# Exit program if no detections have been made
if output.size(0) == 0:
    print ("No detections were made")
    exit()
else:
    pass

# Transform coordinates of the bounding boxes to boundaries of the original image on the padded image
input_dim_list = torch.index_select(input_dim_list, 0, output[:,0].long())
scaling_factor = torch.min(input_dim/input_dim_list,1)[0].view(-1,1)

output[:,[1,3]] -= (input_dim - scaling_factor*input_dim_list[:,0].view(-1,1)) / 2
output[:,[2,4]] -= (input_dim - scaling_factor*input_dim_list[:,1].view(-1,1)) / 2

output[:,1:5] /= scaling_factor # Undo re-scaling in letterbox_image()

# Clip bounding boxes that exceed the image boundaries to the edges of the image
for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, input_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, input_dim_list[i,1])

output_recast = time.time()

# Load colours for bounding boxes
class_load = time.time()
colors = pkl.load(open("pallete", "rb"))

# Draw bounding boxes
draw = time.time()

def draw_boxes(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cl = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cl])
    cv2.rectangle(img, c1, c2, color, 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1) # Create a filled rectangle for class name
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1) # Write class name
    return img

list(map(lambda x: draw_boxes(x, loaded_imgs), output))

# Prefix each image name with 'output_'
det_names = pd.Series(img_list).apply(lambda x: "{}/det_{}".format(args.output,x.split("/")[-1]))

# Write the images with detections to the output folder
list(map(cv2.imwrite, det_names, loaded_imgs))
end = time.time()

# Print time summary
print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", detect_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(img_list)) +  " images)", output_recast - detect_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(img_list)))
print("----------------------------------------------------------")

torch.cuda.empty_cache()