from os import listdir
from os.path import isfile, join
import ntpath
from pathlib import Path
import itertools
# for f in listdir("/Users/ngannguyen/Desktop/keras-frcnn/Apple/0a7e071b30e23d13.jpg"):
#     if isfile(join("/Users/ngannguyen/Desktop/keras-frcnn/Apple", f)):
#         print("/Users/ngannguyen/Desktop/keras-frcnn/Apple"+f+",")
path = "/Users/ngannguyen/Desktop/keras-frcnn/Label_apple/"
txtFilePathList = Path(path).glob('**/*.txt')
image_path = "/Users/ngannguyen/Desktop/keras-frcnn/Apple"


pathlist = Path(image_path).glob('**/*.jpg')

for txtfile in txtFilePathList:
    with open(txtfile, "r+", encoding="utf8") as input:
        lines = input.readlines()
        with open(txtfile, "w+", encoding="utf8") as output:
            for line in lines:
                # print(line)
                output.write(line.replace(" ",","))
    with open(txtfile, "r+", encoding="utf8") as input:
        lines = input.readlines()
        with open(txtfile, "w+", encoding="utf8") as output:
            for line in lines:
                filename_stem = Path(ntpath.basename(input.name)).stem
                img_dir = Path(image_path).glob('**/{}.jpg'.format(filename_stem))
                img = str(next(itertools.islice(img_dir, 0, None)))
                output.write(line.replace("Apple", img))
    with open(txtfile, "r+", encoding="utf8") as input:
        lines = input.readlines()
        with open(txtfile, "w+", encoding="utf8") as output:
            for line in lines:
                output.write(line.strip("\n") + ",apple\n")

output_path = "/Users/ngannguyen/Desktop/keras-frcnn/data/my_data.txt"
txtFilePathList = Path(path).glob('**/*.txt') 

with open(output_path, "w+", encoding="utf8") as output:
    for txtfile in txtFilePathList:
        with open(txtfile, "r+", encoding="utf8") as input:
            lines = input.readlines()
            for line in lines:
                # print(line)
                output.write(line)
print(int(424.959744))

