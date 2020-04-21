# RetinaNet
Keras RetinaNet implementation of RetinaNet.

## Requirements
* numpy
* Keras
* TensorFlow

## Installation
Follow installation instructions for Keras implementation of RetinaNet object detection, under the References section.

## Obtaining CSV data
Skip this step if you have CSV files in the format required by this implementation.
Files in scripts folder are used to create CSV files (class.csv, train.csv, validation.csv, test.csv) from our OID dataset and remove labels without an associated image. Move these files to csv_folder in our dataset and run:
```
python rm-invalid.labels.py ../Dataset
python get-csv.py ../Dataset
```

## Training
Assuming scripts were used to obtain CSV files, in the csv_folder, run:
```
retinanet-train --epoch 2 --steps 453 --batch-size 4 csv train.csv class.csv
```

The training parameters used above are:
* epoch: `--epoch 2`
* steps per epoch: `--steps 453`
* batch size: `--batch-size 4`

## Convert Training Model to Inference Model
To convert training model to inference model, use the template:
```
retinanet-convert-model /path/to/training-model.h5 /path/to/inference-model.h5
```

## Testing
To test the model, run the script `test-model-all.py` on the dataset containing the fruits.
```
python path/to/test-model-all.py path/to/OID/dataset
```
This will print the average AP scores for apples, bananas, and oranges, and the overall AP score over all these fruits.

Running the `test-model-image.py` script will allow you to choose an image to run the model on. You must change the model's name and its path to run the script. Optionally, you can change the score threshold to adjust what boxes will appear in the image.

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
