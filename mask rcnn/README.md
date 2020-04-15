# Mask RCNN

Keras implementation of Mask RCNN for training on custom data.

Keras = 2.2.5.
Tensorflow = 1.x

OIDv4 ToolKit used to download data from Open Images Dataset V5

## References

```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}

@misc{OIDv4_ToolKit,
  title={Toolkit to download and visualize single or multiple classes from the huge Open Images v4 dataset},
  author={Vittorio, Angelo},
  year={2018},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/EscVM/OIDv4_ToolKit}},
}
```

Replace /mrcnn/model.py and /mrcnn/utils.py with the given model.py and utils.py

*[model.py](https://github.com/matterport/Mask_RCNN/pull/1611)

*[util.py](https://github.com/matterport/Mask_RCNN/issues/912)

*[Object detection using Mask R-CNN on a custom dataset](https://towardsdatascience.com/object-detection-using-mask-r-cnn-on-a-custom-dataset-4f79ab692f6d)

*[Splash of Color: Instance Segmentation with Mask R-CNN and TensorFlow](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46)

*[oid_to_pascal_voc_xml.py used to convert labels to xml](https://gist.github.com/nilsfed/1dbf1cf397db50c90705daa6a81a8dec)