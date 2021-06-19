import numpy as np
import pandas as pd
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import threading
import time
import requests

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from keras.backend.tensorflow_backend import set_session  
from imutils.video import FPS
from datetime import datetime
from threading import Timer

# Read webcam
cap = cv2.VideoCapture(0)

config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
set_session(tf.Session(config=config)) 

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables

# What model to download.
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Data Header
header = ['time stamp', 'people count', 'item_count', 'score_avg', 'score_std', 'score_median', 'score_max', 'score_min' , 'min_threshold' ]

# csv file name 
csv_name = 'object_detection_count_results.csv'

# Results in dataframe
df = pd.DataFrame(columns=header)

# set minimum threshold of score
MIN_THRES = 0.5

# get the class id to add into power bi
classnum_add_to_powerbi = 1

# ## Create csv
# if the csv file exists, delete, otherwise create it
try:
    os.remove(csv_name)
    df.to_csv(csv_name, index=False)
except:
    df.to_csv(csv_name, index=False)


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# Threading
push_count = 0
class myThread(threading.Thread):
    def __init__(self, image_np, boxes, classes, scores, min_threshold, fps):
        self.image_np = image_np
        self.boxes = boxes
        self.classes = classes
        self.scores = scores
        self.min_threshold = min_threshold
        self.fps = fps  

    def push_data(self):        
        # Prepare data to Power BI dashboard
        # Count the number of person with scores above the threshold
        show_boxes_all = self.scores > self.min_threshold
        show_boxes = ((self.scores > self.min_threshold) & (self.classes == classnum_add_to_powerbi))
        people_count = len(show_boxes[show_boxes])
        item_count = len(show_boxes_all[show_boxes_all])


        # Format data to send as JSON
        now = datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S%Z")
        if people_count == 0:
            score_avg = 0
            score_std = 0
            score_median = 0
            score_min = 0
            score_max = 0
        else:

            score_avg = np.mean(self.scores[self.scores > self.min_threshold])
            score_std = np.std(self.scores[self.scores > self.min_threshold])
            score_median = np.median(self.scores[self.scores > self.min_threshold])
            score_min = np.min(self.scores[self.scores > self.min_threshold])
            score_max = np.max(self.scores[self.scores > self.min_threshold])

        # data score
        data = [now, item_count, people_count, score_avg, score_std, score_median, score_max, score_min, self.min_threshold]
        
        # Limit the amount of times data can be pushed per second
        max_pushes_per_second = 2
        time_seconds = time.time() % 60
        remainder = (time_seconds - np.floor(time_seconds)) % (1/max_pushes_per_second)

        # Reset count limit every second (leave some jitter room)
        jitter = 0.1
        global push_count
        if (time_seconds - np.floor(time_seconds)) < jitter*2:
            push_count = 0
        
        # If within jitter (0.1 seconds) of the timepoints where we can send data (if max_pushes_per_second = 4, then we can push at 0.25, 0.5, 0.75, and at 0)
        if (remainder <= jitter) or (abs(remainder - (1/max_pushes_per_second)) <= jitter):
            push_count += 1
        if push_count <= max_pushes_per_second:
            try:
                
                # data into data frame
                df_lp = pd.DataFrame([data], columns=header)
    
                # write to the csv file and append
                with open(csv_name, 'a') as f:
                    df_lp.to_csv(f, header=False, index=False)

            #response = requests.post(REST_API_URL, data=binary_data)
            except requests.ConnectionError as e:
                print('[ERROR] Connection Error')
                print(str(e))       


# In[8]:

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      
      fps = FPS().start()

      ret, image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      # Visualization of the results of a detection.
      min_threshold = MIN_THRES

      # Scores
      myThread0 = myThread(image_np, boxes, classes, scores, min_threshold, fps)

      # Push data to data frame
      t2 = threading.Thread(target = myThread0.push_data)
      t2.daemon = True
      t2.start()
    
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          min_score_thresh=min_threshold,
          line_thickness=8)

      # visualize it
      cv2.imshow('object detection', cv2.resize(image_np, (1000,800)))
      #cv2.imshow('object detection', image_np)

      # if press q, it will exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
