# NLP + Neural Network Resume Chatbot

This is a simple resume intent based chatbot which uses Natural Language Processing (NLP) to understand and interpret human languages and uses Tensorflow's neural network to train the model so the model will predict the user's intent and select the best response from the given tag.

## Installation
Create an environment and install the required python libraries (environment.yaml for anaconda and/or requirements.txt for python script). The environment name in this example is "chtbot"


## Scripts

1. [Build Resume chatbot.ipynb](https://github.com/yvien226/Useful-Python-Scripts/blob/master/NLP/Resume%20Chatbot/Build%20Resume%20chatbot.ipynb): The notebook to extract json data from data/ folder, process, train and save the model (/model folder)
2. [chat.py](https://github.com/yvien226/Useful-Python-Scripts/blob/master/NLP/Resume%20Chatbot/chat.py) : loads the model and runs the chatbot

## Steps by step guide
1. Model training
  - Once the environment has been created and all the required python packages have been installed, activate the environment
  - Edit the data/resume_data.json file and enter your own information.
  - Run the "Build Resume chatbot" notebook to train the model, adjust the parameters if needed. The model will be saved in the "model/" folder
2. Run the chatbot
  - Once the environment has been created and all the required python packages have been installed, activate the environment in the cmd by entering `> .\chtbot\Scripts\activate`
  - Run the chat.py script, `> python chat.py`

## Sample Screenshot
![image](https://github.com/yvien226/Useful-Python-Scripts/blob/master/NLP/Resume%20Chatbot/sample_chat.png)

## Python Libraries
- numpy
- tensorflow
- nltk
- flask


