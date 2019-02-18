import ujson as json
import os
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import collections

data_path = os.getcwd()
error_path = os.path.join(data_path, 'data/FALSE.json')
train_path = os.path.join(data_path, 'data/raw_data/altlex_train_bootstrapped.tsv')
test_path = os.path.join(data_path, 'data/raw_data/altlex_gold.tsv')
wnl = WordNetLemmatizer()

with open(error_path, 'r') as fh:
    error_meta = json.load(fh)


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
    with open(path, 'r', encoding='utf8') as fh:
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


def gen_test(path):
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
            if i in error_meta['FP']:
                fp_alt.append(alt)
            elif i in error_meta['FN']:
                fn_alt.append(alt)
            all_alt.append(alt)
    fh.close()
    ids = [j for j in range(1, i+1)]
    for fp in error_meta['FP']:
        ids.remove(fp)
    for fn in error_meta['FN']:
        ids.remove(fn)

    return fp_alt, fn_alt, all_alt, ids


train_all_alt = gen_train(train_path)
train_set = set(train_all_alt)
test_fp_alt, test_fn_alt, test_all_alt, ids = gen_test(test_path)
test_set = set(test_all_alt)
not_in_train = test_set.difference(test_set.intersection(train_set))
print('Not in train: ', not_in_train)

tc = collections.Counter(train_all_alt)
fpc = collections.Counter(test_fp_alt)
fnc = collections.Counter(test_fn_alt)
allc = collections.Counter(test_all_alt)
print('Top 5 in Train: ', tc.most_common(5))
print('Top 5 in Test: ', allc.most_common(5))
print('Top 5 in FP: ', fpc.most_common(5))
print('Top 5 in FN: ', fnc.most_common(5))
print(ids)

# english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%',
#                         '"', '``', '-', '\'\'']
# seg_test_filtered = [[[word.lower() for word in seg if word not in english_punctuations] for seg in sen] for sen in sens]
print('\nhello world!')
