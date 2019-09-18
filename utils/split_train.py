import os
import sys
import random
import pandas as pd


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
    labels, eng_pre, eng_alt, eng_cur, sim_pre, sim_alt, sim_cur = [], [], [], [], [], [], []
    for x in examples:
        labels.append(x[0])
        eng_pre.append(x[1])
        eng_alt.append(x[2])
        eng_cur.append(x[3])
        sim_pre.append(x[4])
        sim_alt.append(x[5])
        sim_cur.append(x[6])

    df = pd.DataFrame({'label': labels, 'eng_pre': eng_pre, 'eng_alt': eng_alt, 'eng_cur': eng_cur,
                       'sim_pre': sim_pre, 'sim_alt': sim_alt, 'sim_cur': sim_cur})
    df.to_csv(os.path.join(file_path, 'train_boot_' + str(prob)[-1]) + '.csv', sep='\t', index=False)


if __name__ == '__main__':
    # train_path = '../data/raw_data'
    # boot_file = 'altlex_train_bootstrapped.tsv'
    # preprocess_train(train_path, boot_file, float(sys.argv[1]))

    path = '../data/raw_data'
    train = 'train_training.csv'
    boot = 'train_boot.csv'
    tr_lines = open(os.path.join(path, train), 'r', encoding='ISO-8859-1').readlines()
    tr_samples = {}
    for idx, line in enumerate(tr_lines):
        line = line.strip().split(',')
        if line[0] == 'labels' and line[1] == 'examples':
            continue
        tr_samples[line[1]] = idx

    bt_lines = open(os.path.join(path, boot), 'r', encoding='ISO-8859-1').readlines()
    bt_samples = {}
    for idx, line in enumerate(bt_lines):
        line = line.strip().split(',')
        if line[0] == 'labels' and line[1] == 'examples':
            continue
        bt_samples[line[1]] = idx

    dif = set(bt_samples.keys()) - set(tr_samples.keys())
    print(len(dif))
    ids = [bt_samples[k] for k in dif]