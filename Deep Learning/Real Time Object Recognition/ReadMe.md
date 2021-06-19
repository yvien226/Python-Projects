# Real Time Object Recognition using Tensorflow

This folder contains machine learning models implemented by researchers in [TensorFlow](https://tensorflow.org). The models are maintained by their respective authors.

I took a tutorial notebook from Tensorflow and made some changes to the code in order to produce real time event object recognition. 

## Installation

You will need to install Tensorflow and a pre-trained model in order to run the code. The pre-trained model that I used is Faster RCNN Inception v2 coco. The scripts are based on Tensorflow 1 and it works on my Tensorflow 1.11 version. However, it is recommended to install Tensorflow 2.
1. [Tensorflow](https://www.tensorflow.org/install)
2. Pre-trained models: [Tensorflow 1 zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md), [Tensorflow 2 zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)


## Scripts

1. [OD_object_detection_tutorial_webcamPY](https://github.com/yvien226/Useful-Python-Scripts/blob/master/Deep%20Learning/Real%20Time%20Object%20Recognition/OD_object_detection_tutorial_webcamPY.py): Real time object recognition via computer webcam using Faster RCNN Inception pre-trained model
2. [OD_object_detection_count_livegraph](https://github.com/yvien226/Useful-Python-Scripts/blob/master/Deep%20Learning/Real%20Time%20Object%20Recognition/OD_object_detection_count_livegraph.py) : Real time object recognition and counting people. The number of people detected during the real time event is recorded in the excel file ( object_detection_count_results.xlsx). The idea is to produce a real time human detection and counting.

## Screenshot
![image](https://user-images.githubusercontent.com/34856605/122636506-fc80d580-d12c-11eb-8abf-2c5fe4ae675a.png)


## Python Libraries
- numpy
- pandas
- openCV
- tensorflow
- matplotlib (not used in the script)
