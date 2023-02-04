
import string
import random 
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer 


def clean_text(text): 

  # initializing lemmatizer to get stem of words
  lemmatizer = WordNetLemmatizer()

  tokens = nltk.word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
  return tokens

def bag_of_words(text, vocab): 
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens: 
    for idx, word in enumerate(vocab):
      if word == w: 
        bow[idx] = 1
  return np.array(bow)

def pred_class(text, vocab, labels, model): 
  # set threshold
  thresh = 0.5
  
  # predict the closest intend
  bow = bag_of_words(text, vocab)
  result = model.predict(np.array([bow]))[0]
  y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
  
  return_list = []
  
  # add fallback if chatbot is unable to determine any relevant intents from the user input
  if len(y_pred) == 0 or np.sum(bow) == 0:
      return_list = ['fallback']
      return return_list
  else:
      y_pred.sort(key=lambda x: x[1], reverse=True)
      
      for r in y_pred:
        return_list.append(labels[r[0]])
      return return_list

def get_response(intents_list, intents_json): 

  tag = intents_list[0]
  list_of_intents = intents_json['intents']
  for i in list_of_intents: 
    if i['tag'] == tag:
      result = random.choice(i['responses'])
      break
  return result