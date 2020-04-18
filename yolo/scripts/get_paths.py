"""
Get paths of all training images and write to a text file
"""

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(dest="dataset", default="train", type=str)

args = parser.parse_args()

image_files = []

path = "../imgs/" + args.dataset + "/"

for filename in os.listdir(path):
    if filename.endswith(".jpg"):
        image_files.append(path + filename)

result = "../data/" + args.dataset + ".txt"

with open(result, "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")