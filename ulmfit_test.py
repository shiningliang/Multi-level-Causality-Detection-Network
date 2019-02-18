import fastai
from fastai import *
from fastai.text import *
import pandas as pd
import numpy as np
from functools import partial
import io
import os
from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.corpus import stopwords

if __name__ == '__main__':
    dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
    documents = dataset.data
    df = pd.DataFrame({'label': dataset.target, 'text': dataset.data})
    print(df.shape)
    df = df[df['label'].isin([1, 10])]
    df = df.reset_index(drop=True)
    print(df['label'].value_counts())

    stop_words = stopwords.words('english')
    df['text'] = df['text'].str.replace("[^a-zA-Z]", " ")
    # tokenization
    tokenized_doc = df['text'].apply(lambda x: x.split())
    # remove stop-words
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
    # de-tokenization
