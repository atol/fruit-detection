#!/usr/bin/env python
# coding: utf-8

# ## Load necessary modules
# import keras
import keras

import sys
sys.path.insert(0, '../')

from tkinter.filedialog import askopenfilename

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import argparse

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

# use this to change which GPU to use
gpu = 0

# set the modified tf session as backend in keras
setup_gpu(gpu)


# ## Load RetinaNet model

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('..', 'my_models', 'inference-model-01.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'Apple', 1: 'Banana', 2: 'Orange'}

# list of fruits
fruits = ["Apple", "Banana", "Orange"]

# parses over directory
parser = argparse.ArgumentParser()
parser.add_argument(dest="dataset", default="test", type=str)

args = parser.parse_args()

# ## Run detection on example

totalFruitScore = 0
totalCount = 0

# load image
for fruit in fruits:    

    fruitScore = 0
    count = 0

    path = args.dataset + "/test/" + fruit + "/"

    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            image = read_image_bgr(path + filename)

            # copy to draw on
            #draw = image.copy()
            #draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

            # preprocess image for network
            image = preprocess_image(image)
            image, scale = resize_image(image)

            # process image
            #start = time.time()
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            #print("processing time: ", time.time() - start)

            # correct for image scale
            #boxes /= scale

            # adds to total score and count
            fruitScore += scores[0][0]
            count += 1

    print("Score of " + fruit + " is: " + str(fruitScore/count))
    totalFruitScore += fruitScore
    totalCount += count

print("Score over all fruits is: " + str(totalFruitScore/totalCount))

    # visualize detections
"""     for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.3:
            break
            
        color = label_color(label)
        
        b = box.astype(int)
        draw_box(draw, b, color=color)
        
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
        
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show() """
