import os
import string
import json
import random 
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer 
import tensorflow as tf 
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Read data from json file
with open('data/resume_data.json') as json_file:
    data = json.load(json_file)

data = data['resume']

# initializing lemmatizer to get stem of words
lemmatizer = WordNetLemmatizer()

# Each list to create
words = []
classes = []
doc_X = []
doc_y = []

# Loop through all the intents
# tokenize each pattern and append tokens to words, the patterns and
# the associated tag to their associated list
for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_y.append(intent['tag'])
    
    # add the tag to the classes if it's not there already 
    if intent['tag'] not in classes:
        classes.append(intent['tag'])
        
# lemmatize all the words in the vocab and convert them to lowercase
# if the words don't appear in punctuation
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

# sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur
words = sorted(set(words))
classes = sorted(set(classes))


# load model
model = tf.keras.models.load_model('model/resume_model')

bot_name = "yeevbot"

# run chatbot
if __name__ == "__main__":
    print("Hi, I'm " + bot_name + " Let's chat! (type 'quit' to exit)")
    while True:
        # user input
        message = input("You: ")
        if message == "quit":
            break

        intents = utils.pred_class(message, words, classes, model)
        result = utils.get_response(intents, data)
        print(result)