# Sewer Pipe Analysis Object Recognition using Tensorflow

This folder contains machine learning models implemented by researchers in [TensorFlow](https://tensorflow.org). The models are maintained by their respective authors.

This is one of the projects I used to work in City West Water as part of the proof of concept. The aim is to assess the video footage of sewerage pipes automatically and identify irregularities. For example: cracks, tree roots and void (hole). 


## Approach

The model has been developed utilising 2 key open souces, Tensorflow and [LabelIMG](https://github.com/tzutalin/labelImg). The outcome was achieved by generating more than 1000 images of labelled defects taken from various CCTV footages. The team at CWW gathered images from existing sewer pipe cctv footage and used LabelIMG to create labels of the observed defects. I then translate the image into a suitable data format and run through the training process using a pre-trained model (Faster RCNN Resnet50) with Tensorflow API.

## Results

The machine learning model generates a reasonably precise and accurate with mAP of 0.40 and a total loss of 0.21. Its strength lies particularly in tree roots while voids are less detectable. Overall there are many ways to improve the model such as performing data augmentation to increase more images with different types of pipe, but this is a good first attempt to automate sewer pipe defects.

Unfortunately I can't share the model as it belongs to the company, but CWW has posted video clip of the outcome via LinkedIn, click [here](https://www.linkedin.com/posts/city-west-water_machinelearning-innovation-activity-6481329039096320000-vPgW/) to watch or see results of video clip below.


## Scripts

1. [OD_object_detection_tutorial_webcamPY](https://github.com/yvien226/Useful-Python-Scripts/blob/master/Deep%20Learning/Real%20Time%20Object%20Recognition/OD_object_detection_tutorial_webcamPY.py): Real time object recognition through computer webcam using Faster RCNN Inception pre-trained model


## Screenshot
![](https://im2.ezgif.com/tmp/ezgif-2-5158d0373a.gif)

## Python Libraries
- numpy
- pandas
- openCV
- tensorflow
- matplotlib (not used in the script)

