"""
Darknet architecture
Based on: https://blog.paperspace.com/tag/series-yolo/
"""

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from utils import *


class EmptyLayer(nn.Module):
    """ Dummy layer for layer operations """
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    """ Layer for holding anchors used to detect bounding boxes """
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

class Darknet(nn.Module):
    """ YOLOv3 object detection model """
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
    
    def forward(self, x, cuda=False):
        outputs = {} # Dictionary storing the output feature maps of every layer
        detections = torch.Tensor()
        
        for idx, module in enumerate(self.blocks[1:]): # Skip over 'net' block
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[idx](x)

            elif module_type == "route":
                layers = [int(x) for x in module["layers"].split(",")]

                # Get relative index
                if layers[0] > 0:
                    layers[0] = layers[0] - idx

                # If layers has only one value, output feature map of the layer indexed by the value
                if len(layers) == 1:
                    x = outputs[layers[0]+idx]

                # If layers has two values, return concatenated feature maps of layers indexed by its values
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - idx

                    x = torch.cat((outputs[layers[0]+idx], outputs[layers[1]+idx]), 1)
            
            elif module_type == "shortcut":
                froms = int(module["from"])
                x = outputs[idx-1] + outputs[idx+froms]
            
            elif module_type == "yolo":
                anchors = self.module_list[idx][0].anchors

                # Get input image dimensions
                img_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(module["classes"])

                # Transform feature map to 2D tensor
                x = predict_transform(x.data, img_dim, anchors, num_classes, cuda)

                # Concatenate x to detections
                detections = torch.cat((detections, x), 1)

            outputs[idx] = x
        
        return detections
    
    def load_weights(self, weightfile):
        """ Load pretrained weights """
        with open(weightfile,"rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5) # First 5 values are header info
            self.header = torch.from_numpy(header) # Save header info for when saving weights
            self.seen = self.header[3] # Number of images seen
            weights = np.fromfile(f, dtype=np.float32) # Read in weights
        
        ptr = 0 # Keep track of position in weights array
        for idx, module in enumerate(self.blocks[1:]):
            # If module type is convolutional, load weights
            if module["type"] == "convolutional":
                conv_layer = self.module_list[idx][0]

                # Check if conv_layer has batch_normalize
                try:
                    batch_normalize = int(module["batch_normalize"])
                except:
                    batch_normalize = 0
                
                # If batch_normalize is True, load weights
                if batch_normalize:
                    bn_layer = self.module_list[idx][1]
                    # Get number of biases
                    num_bn_biases = bn_layer.bias.numel()
                    # Biases
                    bn_biases = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases
                    # Weights
                    bn_weights = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases
                    # Running mean
                    bn_running_mean = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases
                    # Running variance
                    bn_running_var = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases
                    # Reshape to dimensions of bn_layer
                    bn_biases = bn_biases.view_as(bn_layer.bias)
                    bn_weights = bn_weights.view_as(bn_layer.weight)
                    bn_running_mean = bn_running_mean.view_as(bn_layer.running_mean)
                    bn_running_var = bn_running_var.view_as(bn_layer.running_var)
                    # Copy data to bias layer
                    bn_layer.bias.data.copy_(bn_biases)
                    bn_layer.weight.data.copy_(bn_weights)
                    bn_layer.running_mean.data.copy_(bn_running_mean)
                    bn_layer.running_var.data.copy_(bn_running_var)

                # Otherwise, load weights of convolutional layer
                else:
                    num_biases = conv_layer.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr : ptr + num_biases])
                    conv_biases = conv_biases.view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_biases)
                    ptr += num_biases
            
                # Load convolutional layer weights
                num_weights = conv_layer.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights])
                conv_weights = conv_weights.view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_weights)
                ptr += num_weights

def parse_cfg(cfgfile):
    """
    Parses the official cfg file and stores every block as a dict.
    Returns a list of blocks.
    """
    # Read and pre-process the cfg file
    with open(cfgfile,"r") as f:
        lines = f.read().split("\n") # Store lines in a list
        lines = [l for l in lines if len(l) > 0 and l[0] != "#"] # Get rid of empty lines and comments 
        lines = [l.rstrip().lstrip() for l in lines] # Get rid of whitespace
    
    # Separate network blocks
    block = {}
    blocks = []

    for line in lines:
        # Start of new layer
        if line[0] == "[":
            # If block is not empty, it must be storing values of previous layer
            if len(block) != 0:
                blocks.append(block) # Add previous layer to blocks list
                block = {} # Re-initialize block for next layer
            block["type"] = line[1:-1].rstrip() # Get type of block
        # Otherwise, get values from cfg file
        else:
            key, value = line.split("=") 
            block[key.rstrip()] = value.lstrip() # Get rid of whitespace

    blocks.append(block)

    return blocks

def create_modules(blocks):
    """
    Takes a list of blocks returned by parse_cfg() and constructs PyTorch modules
    for the blocks.
    """
    net_info = blocks[0] # First block contains info about network
    module_list = nn.ModuleList() # List containing nn.Module objects
    channels = 3 # Depth of the kernel
    output_filters = [] # List of number of output filters of each block

    # Iterate over the list of blocks and create a PyTorch module for each block
    for idx, block in enumerate(blocks[1:]):
        module = nn.Sequential() # Sequentially execute a number of nn.Module object
        """
        1. Check the type of block
        2. Create a new module for the block
        3. Append to module_list
        """
        # If it's a convolutional layer
        if block["type"] == "convolutional":
            # Get info about the layer
            activation = block["activation"]       
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            padding = (kernel_size - 1) // 2
            
            # Check for batch normalization
            try:
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            
            # Add convolutional layer
            conv = nn.Conv2d(channels, filters, kernel_size, stride, padding, bias=bias)
            module.add_module("conv_{0}".format(idx), conv)

            # Add batch normalization layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(idx), bn)
            
            # Add activation
            # Check for leaky ReLU
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(idx), activn)

        # If it's an upsampling layer
        elif block["type"] == "upsample":
            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="nearest")
            module.add_module("upsample_{0}".format(idx), upsample)

        # If it's a route layer
        elif block["type"] == "route":
            layers = [int(x) for x in block["layers"].split(",")]
            
            # Add route layer
            route = EmptyLayer()
            module.add_module("route_{0}".format(idx), route)

            # Update filters variable to hold the number of filters outputted by a route layer
            filters = sum([output_filters[:][i] for i in layers])
            
        # If it's a shortcut layer (skip connection)
        elif block["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(idx), shortcut)
        
        # If it's the YOLO (detection) layer
        elif block["type"] == "yolo":
            mask = [int(x) for x in block["mask"].split(",")]

            anchors = [int(x) for x in block["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]

            # Select only the anchors indexed by the mask values
            anchors = [anchors[m] for m in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(idx), detection)
    
        module_list.append(module)
        channels = filters
        output_filters.append(filters)
    
    return (net_info, module_list)