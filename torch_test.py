from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
import pandas as pd
import re
import os
import multiprocessing
import logging


wml = WordNetLemmatizer()
SPACE = ' '


def lemmatize_all(sentence):
    wnl = WordNetLemmatizer()
    # sen = [w.lower() for w in word_tokenize(sentence)]
    for word, tag in pos_tag(sentence):
        if tag.startswith('NN'):
            yield wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            yield wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            yield wnl.lemmatize(word, pos='a')
        elif tag.startswith('R'):
            yield wnl.lemmatize(word, pos='r')
        else:
            yield word


def seg_sentence(sentence):
    if sentence == '"\n' or sentence == '\n':
        return []
    sentence = pat_letter.sub('', sentence).strip().lower()
    tokens = word_tokenize(sentence)
    tokens = SPACE.join(lemmatize_all(tokens)).split()
    tokens = [w for w in tokens if w not in stop_words]
    return tokens


if __name__ == '__main__':
    logger = logging.getLogger('Causality')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    path = './data/raw_data/nlp'
    pat_letter = re.compile(r'[^a-zA-Z \']+')
    stop_words = set(stopwords.words('english'))
    # for file_name in os.listdir(path):
    #     logger.info('Reading {}'.format(file_name))
    #     file_path = os.path.join(path, file_name)
    #     lines = open(file_path, 'r', encoding='utf8').readlines()
    #     file_num = file_name.split('_')[0]
    #     if file_num == '2':
    #         out_name = file_num + '_stat.csv'
    #
    #         logger.info('Processing...')
    #         pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count() / 2))
    #         results, sentences = [], []
    #         for line in lines:
    #             results.append(pool.apply_async(seg_sentence, (line,)))
    #         pool.close()
    #         pool.join()
    #         for res in results:
    #             sen = res.get()
    #             if len(sen) > 0:
    #                 sentences.extend(sen)
    #
    #         del results
    #         freq = dict(FreqDist(sentences))
    #         del sentences
    #         words = list(freq.keys())
    #         freqs = list(freq.values())
    #         df = pd.DataFrame({'word': words, 'frequency': freqs})
    #         del words
    #         del freqs
    #         df.to_csv(out_name, sep=',', header=False, index=False)
    #         del df
    file_name = 'merge.csv'
    logger.info('Reading {}'.format(file_name))
    file_path = os.path.join(path, file_name)
    lines = open(file_path, 'r', encoding='utf8').readlines()
    file_num = file_name.split('_')[0]

    logger.info('Processing...')
    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count() / 4))
    results, sentences = [], []
    for line in lines:
        results.append(pool.apply_async(seg_sentence, (line,)))
    pool.close()
    pool.join()
    for res in results:
        sen = res.get()
        if len(sen) > 0:
            sentences.extend(sen)

    del results
    freq = dict(FreqDist(sentences))
    del sentences
    words = list(freq.keys())
    freqs = list(freq.values())
    df = pd.DataFrame({'word': words, 'frequency': freqs})
    del words
    del freqs
    out_name = '7_stat.csv'
    df.to_csv(out_name, sep=',', header=False, index=False)
    del df
