import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from time import time

np.random.seed(int(time()))

SPACE = ' '


def preprocess_train(file_path, file_name, data_type, is_build):
    print("Generating {} examples...".format(data_type))
    examples = []
    engs, sims, labels = [], [], []
    seg_engs, seg_sims = [], []
    data_path = os.path.join(file_path, file_name)
    with open(data_path, 'r', encoding='utf8') as fh:
        for line in tqdm(fh):
            line = line.strip().split('\t')
            labels += [int(line[0])] * 2
            del line[0]
            engs.append(word_tokenize(SPACE.join(line[:3]).strip()))
            sims.append(word_tokenize(SPACE.join(line[3:]).strip()))
            if is_build:
                seg_engs.append([word_tokenize(seg) for seg in line[:3]])
                seg_sims.append([word_tokenize(seg) for seg in line[3:]])
    fh.close()

    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%',
                            '"', '``', '-', '\'\'']
    eng_filtered = [SPACE.join([word.lower() for word in document if word not in english_punctuations])
                    for document in engs]
    sim_filtered = [SPACE.join([word.lower() for word in document if word not in english_punctuations])
                    for document in sims]
    # for i in range(len(eng_filtered)):
    #     t =
    #     examples.append()
    for i in range(len(eng_filtered)):
        examples.append(eng_filtered[i])
        examples.append(sim_filtered[i])

    df = pd.DataFrame({'labels': labels, 'examples': examples})

    return df


def preprocess_test(file_path, file_name, data_type, is_build=False):
    print("Generating {} examples...".format(data_type))
    sentences, segments, labels = [], [], []
    data_path = os.path.join(file_path, file_name)
    with open(data_path, 'r', encoding='ISO-8859-1') as fh:
        for line in tqdm(fh):
            line = line.strip().split('\t')
            num = int(line[-1])
            labels.append(0 if num == 0 else 1)
            sentences.append(word_tokenize(SPACE.join(line[:-1]).strip()))
            if is_build:
                segments.append([word_tokenize(seg) for seg in line[:-1]])
    fh.close()

    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%',
                            '"', '``', '-', '\'\'']
    sen_filtered = [SPACE.join([word.lower() for word in sentence if word not in english_punctuations])
                    for sentence in sentences]
    df = pd.DataFrame({'labels': labels, 'examples': sen_filtered})

    return df


def run_prepare(config, flags):
    df_trn = preprocess_train(config.raw_dir, config.train_file, 'train', config.build)
    # valid_examples, valid_corpus, valid_seg, valid_labels = preprocess_test(config.raw_dir, config.valid_file,
    #                                                                         'valid')
    df_val = preprocess_test(config.raw_dir, config.test_file, 'test', config.build)
    df_trn.to_csv(flags.train_file, index=False)
    df_val.to_csv(flags.test_file, index=False)
