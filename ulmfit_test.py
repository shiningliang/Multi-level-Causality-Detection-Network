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
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
    documents = dataset.data
    df = pd.DataFrame({'label': dataset.target, 'examples': dataset.data})
    print(df.shape)
    df = df[df['label'].isin([1, 10])]
    df = df.reset_index(drop=True)
    print(df['label'].value_counts())

    stop_words = stopwords.words('english')
    df['examples'] = df['examples'].str.replace("[^a-zA-Z]", " ")
    # tokenization
    tokenized_doc = df['examples'].apply(lambda x: x.split())
    # remove stop-words
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
    # de-tokenization
    detokenized_doc = []
    for i in range(len(df)):
        t = ' '.join(tokenized_doc[i])
        detokenized_doc.append(t)

    df['examples'] = detokenized_doc
    # split data into training and validation set
    df_trn, df_val = train_test_split(df, stratify=df['label'], test_size=0.4, random_state=12)
    print(df_trn.shape, df_val.shape)

    # Language model data
    data_lm = TextLMDataBunch.from_df(train_df=df_trn, valid_df=df_val, path='')
    # Classifier model data
    data_clas = TextClasDataBunch.from_df(path="", train_df=df_trn, valid_df=df_val, vocab=data_lm.train_ds.vocab,
                                          bs=32)

    learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.7)
    # train the learner object with learning rate = 1e-2
    learn.fit_one_cycle(10, 1e-2)
    # save encoder
    learn.save_encoder('ft_enc')
    # use the data_clas object we created earlier to build a classifier with our fine-tuned encoder
    learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.7)
    learn.load_encoder('ft_enc')
    learn.fit_one_cycle(10, 1e-2)

    # get predictions
    preds, targets = learn.get_preds()

    predictions = np.argmax(preds, axis=1)
    print(pd.crosstab(predictions, targets))
