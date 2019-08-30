import ujson as json
import os
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import collections
import pandas as pd

wnl = WordNetLemmatizer()


def lemmatize_all(sentence):
    wnl = WordNetLemmatizer()
    sen = [w.lower() for w in word_tokenize(sentence)]
    for word, tag in pos_tag(sen):
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


def gen_train(path):
    all_alt = []
    SPACE = ' '
    i = 0
    with open(path, 'r', encoding='ISO-8859-1') as fh:
        for line in fh:
            i += 1
            line = line.strip().split('\t')
            del line[0]
            alt = SPACE.join(lemmatize_all(line[1]))
            all_alt.append(alt)
            alt = SPACE.join(lemmatize_all(line[4]))
            all_alt.append(alt)
    fh.close()

    return all_alt


def gen_test(path, error_meta=None):
    fp_alt = []
    fn_alt = []
    all_alt = []
    SPACE = ' '
    i = 0
    with open(path, 'r', encoding='utf8') as fh:
        for line in fh:
            i += 1
            line = line.strip().split('\t')
            del line[-1]
            alt = SPACE.join(lemmatize_all(line[1]))
            all_alt.append(alt)
            if error_meta:
                if i in error_meta['FP']:
                    fp_alt.append(alt)
                elif i in error_meta['FN']:
                    fn_alt.append(alt)
    fh.close()
    ids = [j for j in range(1, i + 1)]
    for fp in error_meta['FP']:
        ids.remove(fp)
    for fn in error_meta['FN']:
        ids.remove(fn)

    return fp_alt, fn_alt, all_alt, ids


# english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '"', '``',
# '-', '\'\''] seg_test_filtered = [[[word.lower() for word in seg if word not in english_punctuations] for seg in
# sen] for sen in sens]


if __name__ == "__main__":
    training_path = '../data/raw_data/altlex_train.tsv'
    boot_path = '../data/raw_data/altlex_train_bootstrapped.tsv'
    valid_path = '../data/raw_data/altlex_dev.tsv'
    test_path = '../data/raw_data/altlex_gold.tsv'

    training_all_alt = gen_train(training_path)
    boot_all_alt = gen_train(boot_path)
    training_set = set(training_all_alt)
    boot_set = set(boot_all_alt)

    error_path = '../outputs/bootstrapped/MCDN/results/FALSE_valid.json'
    with open(error_path, 'r') as fh:
        error_meta = json.load(fh)
    test_fp_alt, test_fn_alt, test_all_alt, ids = gen_test(test_path, error_meta)
    test_set = set(test_all_alt)
    not_in_training = test_set.difference(test_set.intersection(training_set))
    not_in_boot = test_set.difference(test_set.intersection(boot_set))
    print("Not in training: ", not_in_training)
    print("Not in bootstrapped: ", not_in_boot)

    tc = collections.Counter(training_all_alt)
    bc = collections.Counter(boot_all_alt)
    fpc = collections.Counter(test_fp_alt)
    fnc = collections.Counter(test_fn_alt)
    tec = collections.Counter(test_all_alt)
    print('Top 5 in Train: ', tc.most_common(5))
    tck, tcv = [], []
    for k, v in tc.most_common(20):
        tck.append(k)
        tcv.append(v)
    print('Top 5 in Test: ', tec.most_common(5))
    tek, tev = [], []
    for k, v in tec.most_common(20):
        tek.append(k)
        tev.append(v)
    print('Top 5 in FP: ', fpc.most_common(5))
    fpk, fpv = [], []
    for k, v in fpc.most_common(20):
        fpk.append(k)
        fpv.append(v)
    print('Top 5 in FN: ', fnc.most_common(5))
    fnk, fnv = [], []
    for k, v in fnc.most_common(20):
        fnk.append(k)
        fnv.append(v)
    print(ids)
    df = pd.DataFrame({'train_word': tck, 'train_freq': tcv, 'test_word': tek, 'test_freq': tev,
                       'fp_word': fpk, 'fp_freq': fpv, 'fn_word': fnk, 'fn_freq': fnv})
    df.to_csv('../analysis.csv', index=False)
