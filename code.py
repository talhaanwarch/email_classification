# -*- coding: utf-8 -*-
"""
Created on Fri May 10 21:09:21 2019

@author: Talha Anwar
"""
import os
import numpy as np
import pickle
import sys
import warnings
from sklearn.externals import joblib
warnings.filterwarnings("ignore")

#with open('text_classifier', 'rb') as training_model:  
#    model = pickle.load(training_model)
   
model = joblib.load('text_classification.pkl')
 
with open('dictionary', 'rb') as f:
    dictionary = pickle.load(f)    
    

#path="E:\\machine learning project\\emaildataset\\emaildataset\\part10\\"
#email_list=  os.listdir(path)

file=sys.argv[1]
features= np.zeros((1, len(dictionary)))
em=open(file)
for line in (em):
    words = line.split()
    for word_index, word in enumerate(dictionary):
        features[0,word_index] = words.count(word)
classifier=model.predict(features)
print(int(classifier))