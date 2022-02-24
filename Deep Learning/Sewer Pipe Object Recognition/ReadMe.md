# Sewer Pipe Analysis Object Recognition using Tensorflow

This folder contains machine learning models implemented by researchers in [TensorFlow](https://tensorflow.org). The models are maintained by their respective authors.

This is a project that I used to work in City West Water as part of the proof of concept. The aim is to assess video footage of sewerage pipes automatically and identify irregularities. For example: cracks, tree roots and void (hole). The purpose of this project is to define a pathway for integrating the model into standard sewer pipe asset management processes and potential ideal future state.


## Approach

The model has been developed utilising 2 key open souces, Tensorflow and [LabelIMG](https://github.com/tzutalin/labelImg). The outcome was achieved by generating more than 1000 images of labelled defects taken from various CCTV footages. The team at CWW gathered images from existing sewer pipe cctv footage and used LabelIMG to create labels of the observed defects. I then translate the image into a suitable data format and run through the training process using a pre-trained model (Faster RCNN Resnet50) with Tensorflow API.

## Results

The machine learning model generates a reasonably precise and accurate with mAP of 0.40 and a total loss of 0.21. Its strength lies particularly in tree roots while voids are less detectable. Overall there are many ways to improve the model such as performing data augmentation to increase more images with different types of pipe, but this is a good first attempt to automate sewer pipe defects.

Unfortunately I can't share the model as it belongs to the company, but CWW has posted video clip of the outcome via LinkedIn, check out the outcome below


## Script

[Object_detection_cctv](https://github.com/yvien226/Useful-Python-Scripts/blob/master/Deep%20Learning/Sewer%20Pipe%20Object%20Recognition/Object_detection_cctv.py): Reads CCTV video footage video file and process with trained deep learning model


## Python Libraries
- numpy
- pandas
- openCV
- tensorflow
- keras


## Outcome (video)

Click [here](https://www.linkedin.com/posts/city-west-water_machinelearning-innovation-activity-6481329039096320000-vPgW/) to check out the original post from City West Water in LinkedIn

![](https://github.com/yvien226/Useful-Python-Scripts/blob/master/Deep%20Learning/Sewer%20Pipe%20Object%20Recognition/Tensorflow%20sewer%20pipe%20analysis.gif)

## Outcome (Images)

The following images are taken from public search engine and process them with the deep learning model. See [Sample Image](https://github.com/yvien226/Useful-Python-Scripts/tree/master/Deep%20Learning/Sewer%20Pipe%20Object%20Recognition/Sample%20Image) and [Sample Image Results](https://github.com/yvien226/Useful-Python-Scripts/tree/master/Deep%20Learning/Sewer%20Pipe%20Object%20Recognition/Sample%20Image%20Results) folders

![](https://github.com/yvien226/Useful-Python-Scripts/blob/master/Deep%20Learning/Sewer%20Pipe%20Object%20Recognition/Sample%20Image%20Results/1_result.jpg)

![](https://github.com/yvien226/Useful-Python-Scripts/blob/master/Deep%20Learning/Sewer%20Pipe%20Object%20Recognition/Sample%20Image%20Results/8_result.jpg)

![](https://github.com/yvien226/Useful-Python-Scripts/blob/master/Deep%20Learning/Sewer%20Pipe%20Object%20Recognition/Sample%20Image%20Results/7_result.jpg)



