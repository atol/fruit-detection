# RetinaNet
Keras RetinaNet implementation of RetinaNet.

## Requirements
* numpy
* Keras
* TensorFlow

## Installation
Follow installation instructions for Keras implementation of RetinaNet object detection, under references.

## Obtaining CSV data
Skip this step if you have CSV files in the format required by this implementation.
Scripts used to create CSV files from our OID dataset (containing categories Apple, Banana, Orange) and remove labels without an associated image. Must move scripts to csv_folder in our dataset and run:
```
python rm-invalid.labels.py ../Dataset
python get-csv.py ../Dataset
```

## Training
Assuming scripts were used to obtain CSV files, in the csv_folder, run:
```
retinanet-train --epoch 2 --steps 453 --batch-size 4 csv train.csv class.csv
```

The training parameters used were:
* Epoch: `--epoch 2`
* Steps: `--steps 453`
* Batch size: `--batch-size 4`

## Convert Training Model to Inference Model
To convert training model to inference model, use the template:
```
retinanet-convert-model /path/to/training/model.h5 /path/to/save/inference/model.h5
```

## Testing
To test the model, use the template:
```
from keras_retinanet.models import load_model
model = load_model('/path/to/model.h5', backbone_name='resnet50')
```

## References
```
@article{RetinaNet,
    title={Focal Loss for Dense Object Detection"}
    author={T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár}
    journal={arXiv}
    publisher={International Conference on Computer Vision}
    year={2017}
    url={https://arxiv.org/abs/1708.02002}
}
```

[Keras implementation of RetinaNet object detection](https://github.com/fizyr/keras-retinanet)