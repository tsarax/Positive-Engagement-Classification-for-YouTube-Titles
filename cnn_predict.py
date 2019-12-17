import json
import pandas as pd
import keras
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import re
import string
import pickle
import gensim

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from keras.backend import set_session


#functions for preprocessing and getting to CNN format

def text_norm(tokenized_sentence):
    """ Takes in a tokenized text to normalize by removing puncuation, numbers, etc. and returns normalized tokens"""
    corpus_normalized=[]
    for document in tokenized_sentence: 
        normalized_sentences=[]
        for i in document:
            #get rid of punct, white space, numbers, etc.
            sent = i.lower() #convert to lowercase
            sent = re.sub(r'\d+', '', sent) 
            sent = sent.translate(str.maketrans('', '', string.punctuation))
            sent = sent.strip() 
            sent = re.sub(r'(?:\n|\s+|\t)', ' ', sent) 
            normalized_sentences.append(sent) 
        normalized_sentences = [i for i in normalized_sentences if i] # remove empty string tokens
        corpus_normalized.append(normalized_sentences) #append sublists all to one list
    return corpus_normalized  #list of lists

def nltk_word_token(d):
    """ Takes in corpus list and returns a list of words within each document (aka list of list). """
    tokenized_word = []
    for document in d:
        word = word_tokenize(document)
        tokenized_word.append(word)
    return tokenized_word

def nltk_stem(tokenized_word):
    """Takes in corpus list and appends stemmed words to a list that is returned. """
    ps = PorterStemmer()
    stem_lists=[]
    for d in tokenized_word:
        stemmed_words=[]
        for word in d: 
            stemmed_words.append(ps.stem(word))
        stem_lists.append(stemmed_words)
    return stem_lists

def token_to_index(token, dictionary):
    """
    Given a token and a gensim dictionary, return the token index
    if in the dictionary, None otherwise.
    Reserve index 0 for padding.
    """
    if token not in dictionary.token2id:
        return None
    return dictionary.token2id[token] + 1

def texts_to_indices(text, dictionary):
    """
    Given a list of tokens (text) and a gensim dictionary, return a list
    of token ids.
    """
    result = list(map(lambda x: token_to_index(x, dictionary), text))
    return list(filter(None, result))


# Load best SVM model
#cnn = load_model('best_cnn.model')
mydict = gensim.corpora.Dictionary.load('cnn.dict')


def make_prediction(text):
    """Takes in text, a video title, and outputs a prediction 
    
    Arguments:
        text {str} -- A video title; can be one word or a sentence
    
    Returns:
        label {str} -- Returns either "above average", "average", or "below average" for the given text
    """
    title = text.strip()
    youtube_title = []
    youtube_title.append(title)

    #prep the reviews we made
    tokenized_texts=nltk_word_token(youtube_title)
    normalized_tokens=text_norm(tokenized_texts)
    stemmed_tokens = nltk_stem(normalized_tokens)

    #prep for CNN
    train_texts_indices = list(map(lambda x: texts_to_indices(x, mydict), stemmed_tokens))
    data = pad_sequences(train_texts_indices, maxlen=12)

    #labels and probability relative to each label
    labels = cnn.predict_classes(data)
    #labels = [int(i) for i in labels]
    #print(labels)


    if labels[0] == 2:
        label = "Above Average"
    if labels[0] == 1:
        label = "Average"
    if labels[0] == 0:
        label= "Below Average"

    return label

    
def load():
    global cnn
    cnn = load_model('best_cnn.model')
    global graph
    graph =  tf.compat.v1.get_default_graph()   
    

#val = input("Please enter the title for the YouTube Video: ")
#load()
#text_test='something here'
#prediction = make_prediction(text_test)
#print(prediction)
