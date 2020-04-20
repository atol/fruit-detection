"""
Writes CSV file with fruit classes.
Writes CSV file with annotation for all labels.
"""

sets = ["train", "test", "validation"]
fruits = ["Apple", "Banana", "Orange"]

import os
import argparse

with open("class.csv", "w") as outfile:
    for i in range(0, len(fruits)):
        outfile.write(fruits[i] + "," + str(i) + "\n")

parser = argparse.ArgumentParser()
parser.add_argument(dest="dataset", default="train", type=str)

args = parser.parse_args()

for st in sets:
    
    txt_files = []

    for fruit in fruits:    

        path = args.dataset + "/" + st + "/" + fruit + "/Label/"

        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                with open(path + filename) as f:
                    for line in f:
                        line = line.split()
                        className, xMin, yMin, xMax, yMax = [str(s) for s in line]
                        csvRow = path[:-6] + filename[:-4] + ".jpg," + xMin.split('.')[0] + "," + yMin.split('.')[0] + "," + xMax.split('.')[0] + "," + yMax.split('.')[0] + "," + className
                        txt_files.append(csvRow)    # appends line in file to csv

    result = st + ".csv"

    with open(result, "w") as outfile:
        for txt in txt_files:
            outfile.write(txt)
            outfile.write("\n")