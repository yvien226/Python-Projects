# Real Time Object Recognition using Tensorflow

This folder contains machine learning models implemented by researchers in [TensorFlow](https://tensorflow.org). The models are maintained by their respective authors.

I took a tutorial notebook from Tensorflow and made some changes to the code in order to produce real time event object recognition. 

## Installation

You will need to install Tensorflow and a pre-trained model in order to run the code. The pre-trained model that I used is Faster RCNN Inception v2 coco. The scripts are based on Tensorflow 1 and it works on my Tensorflow 1.11 version. However, it is recommended to install Tensorflow 2.
1. [Tensorflow](https://www.tensorflow.org/install)
2. Pre-trained models: [Tensorflow 1 zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md), [Tensorflow 2 zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)


## Scripts

1. [OD_object_detection_tutorial_webcamPY](https://github.com/yvien226/Useful-Python-Scripts/blob/master/Deep%20Learning/Real%20Time%20Object%20Recognition/OD_object_detection_tutorial_webcamPY.py): Real time object recognition through computer webcam using Faster RCNN Inception pre-trained model
2. [OD_object_detection_count_livegraph](https://github.com/yvien226/Useful-Python-Scripts/blob/master/Deep%20Learning/Real%20Time%20Object%20Recognition/OD_object_detection_count_livegraph.py) : Real time object recognition and counting people through webcam using the same pre-trained model. The number of people detected during the real time event is recorded live in the excel file ( object_detection_count_results.xlsx). The idea is to produce a real time human detection and counting.

## Screenshot
![image](https://user-images.githubusercontent.com/34856605/122636566-49fd4280-d12d-11eb-8c50-5c057779813d.png)

## Python Libraries
- numpy
- pandas
- openCV
- tensorflow
- matplotlib (not used in the script)
