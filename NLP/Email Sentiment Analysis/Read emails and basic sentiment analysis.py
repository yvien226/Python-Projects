#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import nltk
import spacy
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# outputs
cols = ['subject', 'sender', 'receiver', 'body', 'positive words', 'positive counts', 'negative words', 'negative counts']

output_name_xls ='my_inbox_clean.xlsx'
output_name_csv ='my_inbox_clean.csv'


# read emails data (export outlook inbox emails to csv)
# https://helpdeskgeek.com/office-tips/how-to-export-your-e-mails-from-outlook-to-csv-pst/
data = pd.read_csv("my_inbox.csv", encoding = "ISO-8859-1") 


# remove smtp mail
data = data[data['From: (Type)'] != 'SMTP']
len(data)

# load spacy
spacy_nlp = spacy.load('en_core_web_sm')

# call the VADER object
analyser = SentimentIntensityAnalyzer()


# # Functions

# ### Identify sender and receiver

# sender and receiver name are in this format : lastname, firstname (eg: Smith, John)
# the following function will change to this format: firstname lastname (eg: John Smith)

# identify Sender
def get_sender_name(sender):

    # get the first name and the full name
    if sender.find(",") > 0 :
        sendername = sender.split(", ")[1] + " " + sender.split(", ")[0]
    else:
        sendername = sender
    return sendername


# identify receiver
def get_receiver_name(receiver):
    
    try:

        # if there are multiple receiver
        if receiver.find(";") > 0:
            multi_receiver = receiver.split(";")
            receivername = ""
            for name in multi_receiver:
                if name.find(",") > 0:
                    rname = name.split(", ")[1] + " " + name.split(", ")[0]
                else:
                    rname = name
                receivername = receivername + rname + ", "
            receivername = receivername[:-2]
        else:
            # get the full name
            if receiver.find(",") > 0 :
                receivername = receiver.split(", ")[1] + " " + receiver.split(", ")[0]
            else:
                receivername = receiver
    except:
        receivername = receiver
        
    return receivername


# ### Remove signature 

def remove_sign(body, sender_name):
    # if the sender's first name is in the body, remove it
    length_fnd = body.find(sender_name)
    if length_fnd > 0 :
        body_new = body[:length_fnd]
    else:
        body_new = body
    return body_new


# ### Remove entities
# eg entities: USA, America, *person's name*, day of week, weekday

# remove entities
def remove_entity(nlpdoc):
    doc_noentities = []

    ents = [e.text for e in nlpdoc.ents]
    newString = body_new
    for e in reversed(nlpdoc.ents): #reversed to not modify the offsets of other entities when substituting
        start = e.start_char
        end = start + len(e.text)
        newString = newString[:start] + '' + newString[end:]

    return newString


# # Preprocessing
# Spacy? NLTK? BeautifulSoup for html?
# see: http://ai.intelligentonlinetools.com/ml/sentiment-analysis 
# 
# Removal
# - remove entity (person, city names, geographical places
# - remove stopwords (I, is, you, we, for, and etc)
# - remove punctuation (!,_.?#;)
# - remove spaces 
# - lemmatization
# https://medium.com/@makcedward/nlp-pipeline-stop-words-part-5-d6770df8a936

email_data_list = []

for ind, email in data.iterrows():
    # email data
    subject = email["Subject"]
    body = email["Body"]
    sender = email["From: (Name)"]
    receiver = email["To: (Name)"]
    
    
    # identify sender and receiver
    sender_name = get_sender_name(sender)
    receiver_name = get_receiver_name(receiver)
     # get the first name and the full name
    
    # body text
    # remove signature
    body_new = remove_sign(body, sender_name)
    
    doc = spacy_nlp(body_new)
    
    # remove entities
    body_remove_ent = remove_entity(doc)
    
    doc = spacy_nlp(body_remove_ent)
    
    # lemmatization, remove stop words, spaces and punctuation
    # token.text = original text, token.lemma_ = lemmatization
    tokens = [token.lemma_ for token in doc if not token.is_stop | token.is_punct | token.is_space]
    
    # extract positive and negative words
    pos_word = ""
    neu_word = ""
    neg_word = ""
    pos = 0
    neu = 0
    neg = 0


    # scores each word
    for word in tokens:
        if(analyser.polarity_scores(word)['compound']) >= 0.05:
            pos_word = pos_word + word + ", "
            pos = pos + 1
        elif(analyser.polarity_scores(word)['compound']) <= -0.05:
            neg_word = neg_word + word + ", "
            neg = neg + 1
        else:
            neu_word = neu_word + word + ", "
            neu = neu + 1
            
    # append results
    email_data_list.append([subject, sender_name, receiver_name, body_new, pos_word, pos, neg_word, neg])
    
#dataframe
email_data_df = pd.DataFrame(email_data_list, columns=cols)


# ### save results into excel or csv
email_data_df.to_excel('OUTPUT/' + output_name_xls , index=False)
email_data_df.to_csv('OUTPUT/' + output_name_csv, index=False)




