import numpy as np
import pandas as pd
import spacy as spy
import string
import re
import pickle
import sklearn

def remove_rt(utext):
  utext=utext.replace('rt :','')
  return utext

def remove_punctu(text):
  for punctuation in string.punctuation:
    text=text.replace(punctuation,'')
    return text

def preprocessing(text):
    data=pd.DataFrame([text],columns=['tweet'])
    data['tweet']=data['tweet'].apply(lambda x:x.lower())
    data['tweet']=data['tweet'].str.replace('!','',regex=False)
    data['tweet'] = data['tweet'].apply(lambda x:re.sub(r'@\w+','',x))
    data['tweet']=data['tweet'].apply(remove_rt)
    data['tweet']=data['tweet'].apply(remove_punctu)
    return data['tweet']

with open ('D:\Way to denmark\Projects\Sentiment-Analysis\model\twitter_model.pickle','rb') as f:
    model=pickle.load(f)

