# -*- coding: utf-8 -*-
"""
Created on Fri May 10 21:06:06 2019

@author: Talha Anwar
"""

#libraries
import os
import numpy as np
import re
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import nltk
from collections import Counter
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
   

#stemming lemmatizing stop words initiatiion
ps = PorterStemmer()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

# Creates dictionary from all the emails in the directory
def build_dictionary(dir):
  # Read the file names
  emails = os.listdir(dir)
  emails.sort()
  # Array to hold all the words in the emails
  dictionary = []

  # Collecting all words from those emails
  for email in emails:
    m = open(os.path.join(dir, email))
    for i, line in enumerate(m):
      if i == 2: # Body of email is only 3rd line of text file
        words = line.split()
        dictionary += words

  # We now have the array of words, whoch may have duplicate entries
  dictionary = list(set(dictionary)) # Removes duplicates

  # Removes puctuations and non alphabets
  for index, word in enumerate(dictionary):
    if (word.isalpha() == False) or (len(word) == 1):
      del dictionary[index]
   #remove stopwords 
  for count,word in enumerate(dictionary):
    if word in stop_words:
        del dictionary[count]   
  dictionar_stem=[]
  for w in dictionary:
        dictionar_stem.append(lemmatizer.lemmatize(w.lower()))  
#   dictionary = Counter(dictionar_stem)
#   dictionary = dictionary.most_common(most)
  

  return dictionar_stem

def build_features(dir, dictionary):
  # Read the file names
  emails = os.listdir(dir)
  emails.sort()
  # ndarray to have the features
  features_matrix = np.zeros((len(emails), len(dictionary)))

  # collecting the number of occurances of each of the words in the emails
  for email_index, email in enumerate(emails):
    m = open(os.path.join(dir, email))
    for line_index, line in enumerate(m):
      if line_index == 2: #in each email body is at third line, which make index 2
        words = line.split()
        for word_index, word in enumerate(dictionary):
          features_matrix[email_index, word_index] = words.count(word)

  return features_matrix


def build_labels(dir):
  # Read the file names
  emails = os.listdir(dir)
  emails.sort()
  # ndarray of labels
  labels_matrix = np.zeros(len(emails))

  for index, email in enumerate(emails):
    labels_matrix[index] = 1 if re.search('spms*', email) else 0

  return labels_matrix