# Darknet - Base architecture of YOLO
# Based on the following tutorial:
# https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/


from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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
    for index, block in enumerate(blocks[1:]):
        module = nn.Sequential() # Sequentially execute a number of nn.Module object

        """
        1. Check the type of block
        2. Create a new module for the block
        3. Append to module_list
        """

        # If it"s a convolutional layer
        if block["type"] == "convolutional":
            # Get info about the layer
            activation = block["activation"]       
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            pad = int(block["pad"])

            # Check for padding
            if pad:
                padding = (kernel_size - 1) // 2
            else:
                padding = 0
            
            # Check for batch normalization
            try:
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            
            # Add convolutional layer
            conv = nn.Conv2d(channels, filters, kernel_size, stride, padding, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # Add batch normalization layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
            
            # Add activation
            # Check for either linear ReLU or leaky ReLU
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

        # If it"s an upsampling layer
        elif block["type"] == "upsample":
            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="bilinear")
            module.add_module("upsample_{0}".format(index), upsample)

        # If it's a route layer
        elif block["type"] == "route":
            block["layers"] = block["layers"].split(",")
            # Start of route
            start = int(block["layers"][0])
            # End of route
            try:
                end = int(block["layers"][1])
            except:
                end = 0

            # If indices are positive, get relative indices
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            # Add route layer
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            # Updates filters variable to hold the number of filters outputted by a route layer
            if end < 0:
                # Concatenate feature maps
                filters = output_filters[start+index] + output_filters[end+index]
            else:
                filters = output_filters[start+index]
            
        # If it's a shortcut layer (skip connection)
        elif block["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(index), shortcut)
        
        # If it's the YOLO (detection) layer
        elif block["type"] == "yolo":
            mask = block["mask"].split(",")
            mask = [int(m) for m in mask]

            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]

            # Select only the anchors indexed by the mask values
            anchors = [anchors[m] for m in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
    
        module_list.append(module)
        channels = filters
        output_filters.append(filters)
    
    return (net_info, module_list)


# Define dummy layer for layer operations, e.g. addition or concatenation
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

# Define detection layer for holding anchors used to detect bounding boxes
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

blocks = parse_cfg("cfg/yolov3.cfg")
print(create_modules(blocks))