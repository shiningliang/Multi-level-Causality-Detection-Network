import os
import pickle as pkl
from tqdm import tqdm
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import ujson as json
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from time import time

np.random.seed(int(time()))

SPACE = ' '


def stat(seq_length):
    print('Seq len info :')
    seq_len = np.asarray(seq_length)
    idx = np.arange(0, len(seq_len), dtype=np.int32)
    print(stats.describe(seq_len))
    plt.figure(figsize=(16, 9))
    plt.subplot(121)
    plt.plot(idx[:], seq_len[:], 'ro')
    plt.grid(True)
    plt.xlabel('index')
    plt.ylabel('seq_len')
    plt.title('Scatter Plot')

    plt.subplot(122)
    plt.hist(seq_len, bins=10, label=['seq_len'])
    plt.grid(True)
    plt.xlabel('seq_len')
    plt.ylabel('freq')
    plt.title('Histogram')
    plt.show()


def preprocess_train(data_path, data_type):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    engs, sims, labels = [], [], []
    data_path = os.path.join(data_path, 'altlex_train_bootstrapped.tsv')
    with open(data_path, 'r', encoding='utf8') as fh:
        for line in fh:
            line = line.strip().split('\t')
            labels.append(int(line[0]))
            eng = word_tokenize(SPACE.join(line[1:4]).strip())
            sim = word_tokenize(SPACE.join(line[4:]).strip())
            engs.append(eng)
            sims.append(sim)
    fh.close()

    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%',
                            '"', '``', '-', '\'\'']
    eng_filtered = [[word.lower() for word in document if word not in english_punctuations] for document in engs]
    sim_filtered = [[word.lower() for word in document if word not in english_punctuations] for document in sims]
    total = 0
    seq_len = []
    for label, eng, sim in zip(labels, eng_filtered, sim_filtered):
        # if label == 0:
        #     prob = np.random.random(1)
        #     if prob <= 0.75:
        #         continue
        total += 1
        examples.append({'eid': total,
                         'tokens': eng,
                         'label': label})
        eval_examples[total] = label
        seq_len.append(len(eng))

        total += 1
        examples.append({'eid': total,
                         'tokens': sim,
                         'label': label})
        eval_examples[total] = label
        seq_len.append(len(sim))
    # stat(seq_len)
    # print('Get {} total examples'.format(total))
    # print('Get {} causal examples'.format(causal))
    # print('Get {} non-causal examples'.format(non_causal))
    eng_sentences = [SPACE.join(tokens) for tokens in eng_filtered]
    sim_sentences = [SPACE.join(tokens) for tokens in sim_filtered]
    np.random.shuffle(examples)

    return examples, eval_examples, eng_sentences + sim_sentences, max(seq_len)


def preprocess_test(data_path, data_type):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    sentences, labels = [], []
    data_path = os.path.join(data_path, 'altlex_gold.tsv')
    with open(data_path, 'r', encoding='utf8') as fh:
        for line in fh:
            line = line.strip().split('\t')
            num = int(line[-1])
            labels.append(0 if num == 0 else 1)
            tmp = word_tokenize(SPACE.join(line[:-1]).strip())
            sentences.append(tmp)
    fh.close()

    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%',
                            '"', '``', '-', '\'\'']
    sen_filtered = [[word.lower() for word in sentence if word not in english_punctuations] for sentence in sentences]
    total = 0
    for label, sen in zip(labels, sen_filtered):
        total += 1
        examples.append({'eid': total,
                         'tokens': sen,
                         'label': label})
        eval_examples[total] = label
        # if label == 0:
        #     non_causal += 1
        # else:
        #     causal += 1
    # stat(seq_len)
    # print('Get {} total examples'.format(total))
    # print('Get {} causal examples'.format(causal))
    # print('Get {} non-causal examples'.format(non_causal))
    eng_sentences = [SPACE.join(tokens) for tokens in sen_filtered]

    return examples, eval_examples, eng_sentences


def build_dict(data_path):
    dictionary = {}
    with open(data_path, 'r', encoding='utf8') as fh:
        for line in fh:
            line = line.strip().split(' ')
            fredist = nltk.FreqDist(line)
            for localkey in fredist.keys():
                if localkey in dictionary.keys():
                    dictionary[localkey] = dictionary[localkey] + fredist[localkey]
                else:
                    # 如果字典中不存在
                    dictionary[localkey] = fredist[localkey]  # 将当前词频添加到字典中

    return set(dictionary)


def save(filename, obj, message=None):
    if message is not None:
        print('Saving {}...'.format(message))
        if message == 'corpus':
            with open(filename, 'a', encoding='utf8') as fh:
                fh.writelines([line + '\n' for line in obj])
        else:
            with open(filename, 'w', encoding='utf8') as fh:
                json.dump(obj, fh)


def get_embedding(data_type, corpus_dict, emb_file=None, vec_size=None, token2id_dict=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = set()
    trained_embeddings = {}
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, 'rb') as fin:
            trained_embeddings = pkl.load(fin)
        fin.close()
        embedding_dict = set(trained_embeddings)

    filtered_tokens = corpus_dict.intersection(embedding_dict)  # common
    NULL = "<NULL>"
    OOV = "<OOV>"
    token2id = {token: idx for idx, token in
                enumerate(filtered_tokens, 2)} if token2id_dict is None else token2id_dict
    token2id[NULL] = 0
    token2id[OOV] = 1
    embedding_mat = np.zeros([len(token2id), vec_size])
    for token in filtered_tokens:
        embedding_mat[token2id[token]] = trained_embeddings[token]
    return embedding_mat, token2id


def build_features(samples, data_type, max_len, out_file, word2id):
    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    meta = {}
    for sample in tqdm(samples):
        total += 1
        token_ids = np.zeros([max_len], dtype=np.int32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2id:
                    return word2id[each]
            return 1

        seq_len = min(len(sample['tokens']), max_len)
        for i in range(seq_len):
            token_ids[i] = _get_word(sample['tokens'][i])

        record = tf.train.Example(features=tf.train.Features(feature={
            'eid': tf.train.Feature(int64_list=tf.train.Int64List(value=[sample['eid']])),
            'token_ids': tf.train.Feature(bytes_list=tf.train.BytesList(value=[token_ids.tostring()])),
            'token_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_len])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[sample['label']])),
        }))
        writer.write(record.SerializeToString())
    print('Build {} instances of features in total'.format(total))
    meta['total'] = total
    writer.close()
    return meta


def run_prepare(config, flags):
    train_examples, train_eval_examples, train_corpus, max_len = preprocess_train(config.raw_dir, 'train')
    test_examples, test_eval_examples, test_corpus = preprocess_test(config.raw_dir, 'test')

    # save(flags.corpus_file, train_corpus + test_corpus, 'corpus')
    # corpus_dict = build_dict(flags.corpus_file)
    token2id = None
    flag = True
    if os.path.isfile(flags.token2id_file):
        flag = False
        with open(flags.token2id_file, 'r') as fh:
            token2id = json.load(fh)
    # token_emb_mat, token2id = get_embedding('word', corpus_dict, emb_file=flags.w2v_file, vec_size=config.embed_size,
    #                                         token2id_dict=token2id)
    # save(flags.token_emb_file, token_emb_mat, message='token embedding')
    # if flag:
    #     save(flags.token2id_file, token2id, message='word2idx')
    # del token_emb_mat

    train_meta = build_features(train_examples, 'train', 200, flags.train_record_file, token2id)
    save(flags.train_eval_file, train_eval_examples, message='train eval')
    save(flags.train_meta, train_meta, message='train meta')
    del train_examples, train_eval_examples, train_corpus

    test_meta = build_features(test_examples, 'test', 200, flags.test_record_file, token2id)
    save(flags.test_eval_file, test_eval_examples, message='test eval')
    save(flags.test_meta, test_meta, message='test meta')
    del test_examples, test_eval_examples, test_corpus

    save(flags.shape_meta, {'max_len': 200}, message='shape meta')

# if __name__ == '__main__':
#     root = os.getcwd()
#     train_file = os.path.join(root, 'data/raw_data/altlex_train_bootstrapped.tsv')
#     train_examples, train_eval_examples, train_corpus, max_len = preprocess_train(train_file, 'train')
#     # train_meta = build_features(train_examples, max_len, 'train', flags.train_record_file, token2id)
#     # save(flags.train_eval_file, train_eval_examples, message='train eval')
#     # save(flags.train_meta, train_meta, message='train meta')
#
#     test_file = os.path.join(root, 'data/raw_data/altlex_gold.tsv')
#     test_examples, test_eval_examples, test_corpus = preprocess_test(test_file, 'test')
#     # test_meta = build_features(train_examples, max_len, 'train', flags.test_record_file, token2id)
#     # save(flags.test_eval_file, test_eval_examples, message='test eval')
#     # save(flags.test_meta, test_meta, message='test meta')
#     save(os.path.join(root, 'data/processed_data/corpus.txt'), train_corpus + test_corpus, 'corpus')
