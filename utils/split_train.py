import os
import random


def preprocess_train(file_path, file_name, prob):
    pos_examples = []
    neg_examples = []
    in_path = os.path.join(file_path, file_name)
    lines = open(in_path, 'r', encoding='ISO-8859-1').readlines()
    for line in lines:
        line = line.strip().split('\t')
        label = int(line[0])
        if label == 0:
            tmp = round(random.random(), 4)
            if tmp <= prob:
                neg_examples.append(line)
        elif label == 1:
            tmp = round(random.random(), 4)
            if tmp <= prob:
                pos_examples.append(line)

    examples = pos_examples + neg_examples
    print(len(examples))


train_path = '../data/raw_data'
boot_file = 'altlex_train_bootstrapped.tsv'
preprocess_train(train_path, boot_file, 0.2)
