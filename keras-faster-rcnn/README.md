# keras-frcnn
Keras implementation of Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.
cloned from https://github.com/yhenon/keras-frcnn/




USAGE:
- Both theano and tensorflow backends are supported. However compile times are very high in theano, and tensorflow is highly recommended.
- `train_frcnn.py` can be used to train a model. To train on Pascal VOC data, simply do:
`python train_frcnn.py -p /path/to/pascalvoc/`. 
- the Pascal VOC data set (images and annotations for bounding boxes around the classified objects) can be obtained from: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
- simple_parser.py provides an alternative way to input data, using a text file. Simply provide a text file, with each
line containing:

    `filepath,x1,y1,x2,y2,class_name`

    For example:

    /data/imgs/img_001.jpg,837,346,981,456,cow
    
    /data/imgs/img_002.jpg,215,312,279,391,cat

    The classes will be inferred from the file. To use the simple parser instead of the default pascal voc style parser,
    use the command line option `-o simple`. For example `python train_frcnn.py -o simple -p my_data.txt`.

- Running `train_frcnn.py` will write weights to disk to an hdf5 file, as well as all the setting of the training run to a `pickle` file. These
settings can then be loaded by `test_frcnn.py` for any testing.

- test_frcnn.py can be used to perform inference, given pretrained weights and a config file. Specify a path to the folder containing
images:
    `python test_frcnn.py -p /path/to/test_data/`
- Data augmentation can be applied by specifying `--hf` for horizontal flips, `--vf` for vertical flips and `--rot` for 90 degree rotations

# Reference

https://github.com/kbardool/keras-frcnn


https://www.analyticsvidhya.com/blog/2018/11/implementation-faster-r-cnn-python-object-detection/?fbclid=IwAR2eJVFG9I7YLi3OnIFc6xj0h-MGfj-n8PG7yc80TYVmT_QQjQz4jbScLgQ
