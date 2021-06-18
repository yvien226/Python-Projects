# Urban Sound Recognition

The main objective of this project is to identify the sound events from the urban area using Machine Learning with Netural Networks. 

The data set is taken from the UrbanSound8K(https://urbansounddataset.weebly.com/urbansound8k.html). This dataset contains 8732 labelled sound excepts of urban sounds from 10 classes:
- air conditioner
- car horn
- children playing
- dog bark
- drilling
- engine idling
- gun shot
- jackhammer
- siren
- street music

The work includes the following notebooks:
1. UrbanSound8k Data Exploration and Visualisation: Explore the data sets 
2. Data Preprocessing: Data preparation using Librosa to normalise the audio channel, sample rate and bit depth of the audio file for consistency. 
3. CNN/MLP Building model, training and evaluation: Develop a basic Convolutional Neural Network (CNN) and Multilayer Perceptron (MLP) model, train and evaluate the data sets using keras and scikit-learn
4. CNN model test: Test the CNN model with untrained audio files
5. Real Time CNN model test: Real time sound event recognition with CNN model via computer microphone. The script produces all class prediction probability chart during the real time event.



## Python Libraries
- numpy
- pandas
- seaborn
- matplotlib
- struct
- keras
- scikit-learn
- pyaudio
- librosa
- itertools
