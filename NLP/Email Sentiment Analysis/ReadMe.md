# Email Sentiment Analysis

The original intention of this project is to perform sentiment analysis on my work emails to understand if the message I received are positive, negative or neutral. Sentiment analysis uses natural language processing to recognise the emotional tone behind words. 

## Script
Python Script: [Read emails and basic sentiment analysis](https://github.com/yvien226/Useful-Python-Scripts/blob/master/NLP/Email%20Sentiment%20Analysis/Read%20emails%20and%20basic%20sentiment%20analysis.py)

## Steps
1. Download email inbox data from outlook and save into csv file
2. Read emails and perform data preprocessing, this includes:
    - Remove smtp mails
    - Remove signatures
    - Identify sender and receiver
3. Text Processing: Using Spacy and NLTK to process the content of the email:
    - Remove entities (city names, person's name, geographical places)
    - Remove stopwords (I, is, you, we, for, and etc)
    - Remove punctuation (!,.-?#:)
    - Remove spaces
    - Lemmatisation (Grouping similar words into single word
4. Perform sentiment analysis on each word using [VADER (Valence Aware Dictionary for Sentiment Reasoning)](https://github.com/cjhutto/vaderSentiment) and count the number of positive, neutral and negative words in each email content.
5. Save results into csv/excel file and visualise the data using Power BI.

## Python Libraries
- pandas
- nltk
- spacy
- vaderSentiment


