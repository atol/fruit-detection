"""
Removes labels that do not have a corresponding image in the directory
"""

sets = ["train", "test", "validation"]
fruits = ["Apple", "Banana", "Orange"]

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(dest="dataset", default="train", type=str)

args = parser.parse_args()

for st in sets:
    for fruit in fruits:

        path = args.dataset + "/" + st + "/" + fruit + "/Label/"

        numRmFiles = 0
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                imagePath = path[:-6] + filename[:-4] + ".jpg"
                if not os.path.exists(imagePath):
                    numRmFiles += 1
                    os.remove(path + filename)

        print(str(numRmFiles) + " files removed from " + path)