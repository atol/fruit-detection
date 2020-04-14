from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


def predict_transform(prediction, input_dim, anchors, num_classes, cuda):
    """ Takes a detection feature map and turns it into a 2-D tensor """
    batch_size = prediction.size(0)
    grid_size = prediction.size(2)
    stride = input_dim // grid_size
    bbox_attrs = 5 + num_classes # (x, y, w, h, objectness) + class probabilities
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # Toggle CUDA
    if cuda:
        FloatTensor = torch.cuda.FloatTensor
    else:
        FloatTensor = torch.FloatTensor

    # Use sigmoid function on x,y coordinates to constrain x,y offsets between 0 and 1
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])

    # Use sigmoid function on objectness score to get a probability
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # Apply sigmoid activation to class scores
    prediction[:,:,5:] = torch.sigmoid((prediction[:,:,5:]))

    # Add grid offsets to the center coordinates prediction
    grid = np.arange(grid_size)
    grid_x, grid_y = np.meshgrid(grid, grid)

    x_offset = FloatTensor(grid_x).view(-1,1)
    y_offset = FloatTensor(grid_y).view(-1,1)
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    # Divide anchors by the stride of the detection feature map
    anchors = FloatTensor([(a_w/stride, a_h/stride) for a_w, a_h in anchors])

    # Apply anchors to dimensions of bounding box
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    # Resize the detections map to the size of the input image
    prediction[:,:,:4] *= stride

    return prediction