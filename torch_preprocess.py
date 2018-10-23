import os
import pickle as pkl
from tqdm import tqdm
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import ujson as json
import tensorflow as tf
import torch
import nltk
from nltk.tokenize import word_tokenize
from time import time

np.random.seed(int(time()))

SPACE = ' '


def stat_length(seq_length):
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


def stat_altlex(eng_sentences, sim_sentences, labels):
    c_alt, nc_alt = [], []
    for eng, sim, label in zip(eng_sentences, sim_sentences, labels):
        if label == 0:
            nc_alt.append(' '.join(w for w in eng[1]))
            nc_alt.append(' '.join(w for w in sim[1]))
        else:
            c_alt.append(' '.join(w for w in eng[1]))
            c_alt.append(' '.join(w for w in sim[1]))
    c_alt_set = set(c_alt)
    nc_alt_set = set(nc_alt)
    co_alt_set = c_alt_set.intersection(nc_alt_set)
    co_in_c, co_in_nc = 0, 0
    for c, nc in zip(c_alt, nc_alt):
        if c in co_alt_set:
            co_in_c += 1
        if nc in nc_alt_set:
            co_in_nc += 1
    print('#Altlexes rep casual - {}'.format(len(c_alt_set)))
    print('#Altlexes rep non_casual - {}'.format(len(nc_alt_set)))
    print('#Altlexes in both set - {}'.format(len(co_alt_set)))
    print(co_alt_set)
    print('#CoAltlex in causal - {}'.format(co_in_c))
    print('#CoAltlex in non_causal - {}'.format(co_in_nc))


def seg_length(sentences):
    seg_len = []
    for sen in sentences:
        seg_len.append((len(sen[0]), len(sen[1]), len(sen[2])))
    return seg_len


def preprocess_train(file_path, file_name, data_type, is_build):
    print("Generating {} examples...".format(data_type))
    examples = []
    engs, sims, labels = [], [], []
    seg_engs, seg_sims = [], []
    data_path = os.path.join(file_path, file_name)
    with open(data_path, 'r', encoding='utf8') as fh:
        for line in fh:
            line = line.strip().split('\t')
            labels.append(int(line[0]))
            del line[0]
            engs.append(word_tokenize(SPACE.join(line[:3]).strip()))
            sims.append(word_tokenize(SPACE.join(line[3:]).strip()))
            if is_build:
                seg_engs.append([word_tokenize(seg) for seg in line[:3]])
                seg_sims.append([word_tokenize(seg) for seg in line[3:]])
    fh.close()

    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%',
                            '"', '``', '-', '\'\'']
    eng_filtered = [[word.lower() for word in document if word not in english_punctuations] for document in engs]
    sim_filtered = [[word.lower() for word in document if word not in english_punctuations] for document in sims]
    if is_build:
        seg_eng_filtered = [[[word.lower() for word in seg if word not in english_punctuations] for seg in eng] for eng
                            in seg_engs]
        seg_sim_filtered = [[[word.lower() for word in seg if word not in english_punctuations] for seg in sim] for sim
                            in seg_sims]
    else:
        seg_eng_filtered, seg_sim_filtered = [], []
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
                         'cau_label': label})
        seq_len.append(len(eng))

        total += 1
        examples.append({'eid': total,
                         'tokens': sim,
                         'cau_label': label})
        seq_len.append(len(sim))
    # stat_length(seq_len)
    # print('Get {} total examples'.format(total))
    # print('Get {} causal examples'.format(causal))
    # print('Get {} non-causal examples'.format(non_causal))
    if is_build:
        sentences = []
        for eng_tokens, sim_tokens in zip(eng_filtered, sim_filtered):
            sentences.append(SPACE.join(eng_tokens))
            sentences.append(SPACE.join(sim_tokens))
    else:
        sentences = []
    np.random.shuffle(examples)
    return examples, sentences, (seg_eng_filtered, seg_sim_filtered), labels, max(seq_len)


def preprocess_test(file_path, file_name, data_type, is_build=False):
    print("Generating {} examples...".format(data_type))
    examples = []
    sentences, segments, labels = [], [], []
    data_path = os.path.join(file_path, file_name)
    with open(data_path, 'r', encoding='ISO-8859-1') as fh:
        for line in fh:
            line = line.strip().split('\t')
            num = int(line[-1])
            labels.append(0 if num == 0 else 1)
            sentences.append(word_tokenize(SPACE.join(line[:-1]).strip()))
            if is_build:
                segments.append([word_tokenize(seg) for seg in line[:-1]])
    fh.close()

    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%',
                            '"', '``', '-', '\'\'']
    sen_filtered = [[word.lower() for word in sentence if word not in english_punctuations] for sentence in sentences]
    if is_build:
        seg_filtered = [[[word.lower() for word in seg if word not in english_punctuations] for seg in eng] for eng
                        in segments]
    else:
        seg_filtered = []
    total = 0
    for label, sen in zip(labels, sen_filtered):
        total += 1
        examples.append({'eid': total,
                         'tokens': sen,
                         'cau_label': label})
        # if label == 0:
        #     non_causal += 1
        # else:
        #     causal += 1
    # stat(seq_len)
    # print('Get {} total examples'.format(total))
    # print('Get {} causal examples'.format(causal))
    # print('Get {} non-causal examples'.format(non_causal))
    if is_build:
        sentences = [SPACE.join(tokens) for tokens in sen_filtered]
    else:
        sentences = []
    return examples, sentences, seg_filtered, labels


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
            with open(filename, 'w', encoding='utf8') as fh:
                fh.writelines([line + '\n' for line in obj])
        elif message == 'embeddings':
            with open(filename, 'wb') as fh:
                pkl.dump(obj, fh)
        else:
            with open(filename, 'w', encoding='utf8') as fh:
                json.dump(obj, fh)
        fh.close()


def get_embedding(data_type, corpus_dict, emb_file=None, vec_size=None):
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
    oov_tokens = corpus_dict.difference(filtered_tokens)
    combined_tokens = []
    for token in oov_tokens:
        if len(token.split('-')) > 1:
            combined_tokens.append(token)
    combined_tokens = set(combined_tokens)
    oov_tokens = oov_tokens.difference(combined_tokens)
    print('Filtered_tokens: {} Combined_tokens: {} OOV_tokens: {}'.format(len(filtered_tokens), len(combined_tokens),
                                                                          len(oov_tokens)))
    NULL = "<NULL>"
    OOV = "<OOV>"
    token2id = {token: idx for idx, token in enumerate(filtered_tokens, 2)}
    token2id[NULL] = 0
    token2id[OOV] = 1
    embedding_mat = np.zeros([len(token2id), vec_size])
    for token in filtered_tokens:
        embedding_mat[token2id[token]] = trained_embeddings[token]
    token_tail = len(token2id)
    for token in combined_tokens:
        tokens = token.split('-')
        token_vec = np.zeros([vec_size])
        in_emb = 0
        for t in tokens:
            if t in filtered_tokens:
                token_vec += trained_embeddings[t]
                in_emb += 1
        if in_emb == 0:
            continue
        token2id[token] = token_tail
        embedding_mat = np.row_stack((embedding_mat, token_vec / in_emb))
        token_tail += 1
    embedding_mat[1] = np.random.uniform(-0.25, 0.25)
    return embedding_mat, token2id


def seg_length(sentences):
    seg_len = []
    for sen in sentences:
        seg_len.append((len(sen[0]), len(sen[1]), len(sen[2])))
    return seg_len


def gen_annotation(segs, max_length, filename, labels, data_type):
    if data_type == 'train':
        eng_length = seg_length(segs[0])
        sim_length = seg_length(segs[1])
        with open(filename, 'w', encoding='utf8') as f:
            for el, sl, label in zip(eng_length, sim_length, labels):
                pre, alt, cur = el
                if sum(el) > max_length:
                    cur -= pre + alt + cur - max_length
                annos = '0 ' * pre
                annos += '1 ' if label == 1 else '2 ' * alt
                annos += '0 ' * cur
                f.write(annos.strip() + '\n')
                pre, alt, cur = sl
                if sum(sl) > max_length:
                    cur -= pre + alt + cur - max_length
                annos = '0 ' * pre
                annos += '1 ' if label == 1 else '2 ' * alt
                annos += '0 ' * cur
                f.write(annos.strip() + '\n')
        f.close()
    else:
        length = seg_length(segs)
        with open(filename, 'w', encoding='utf8') as f:
            for l, label in zip(length, labels):
                pre, alt, cur = l
                if sum(l) > max_length:
                    cur -= pre + alt + cur - max_length
                annos = '0 ' * pre
                annos += '1 ' if label == 1 else '2 ' * alt
                annos += '0 ' * cur
                f.write(annos.strip() + '\n')
        f.close()


def build_features(sentences, data_type, max_len, out_file, word2id, annotation_file=None):
    print("Processing {} examples...".format(data_type))
    total = 0
    meta = {}
    samples = []
    fh = open(annotation_file, 'r', encoding='utf8')
    for sentence in tqdm(sentences):
        total += 1
        token_ids = np.zeros([max_len], dtype=np.int32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2id:
                    return word2id[each]
            return 1

        seq_len = min(len(sentence['tokens']), max_len)
        for i in range(seq_len):
            token_ids[i] = _get_word(sentence['tokens'][i])
        samples.append({'id': sentence['eid'],
                        'tokens': token_ids,
                        'length': seq_len,
                        'cau_label': sentence['cau_label']})
    fh.close()
    with open(out_file, 'wb') as fo:
        pkl.dump(samples, fo)
    fo.close()
    print('Build {} instances of features in total'.format(total))
    meta['total'] = total
    return meta


def run_prepare(config, flags):
    train_examples, train_corpus, train_seg, train_labels, max_len = preprocess_train(config.raw_dir,
                                                                                      config.train_file,
                                                                                      'train',
                                                                                      config.build)
    # valid_examples, valid_corpus, valid_seg, valid_labels = preprocess_test(config.raw_dir, config.valid_file, 'valid')
    test_examples, test_corpus, test_seg, test_labels = preprocess_test(config.raw_dir, config.test_file,
                                                                        'test', config.build)
    if config.build:
        types = ['train', 'test']
        labels = [train_labels, test_labels]
        segs = [train_seg, test_seg]
        for t, s, l in zip(types, segs, labels):
            save(os.path.join(config.processed_dir, t + '_corpus.txt'), train_corpus, 'corpus')
            gen_annotation(s, config.max_len, os.path.join(config.processed_dir, t + '_annotations.txt'), l, t)
        corpus_dict = build_dict(flags.corpus_file)
        token_emb_mat, token2id = get_embedding('word', corpus_dict, flags.w2v_file, config.n_emb)
        save(flags.token_emb_file, token_emb_mat, message='embeddings')
        save(flags.token2id_file, token2id, message='word2index')
    else:
        with open(flags.token2id_file, 'r') as fh:
            token2id = json.load(fh)

    train_meta = build_features(train_examples, 'train', config.max_len, flags.train_record_file, token2id,
                                flags.train_annotation)
    save(flags.train_meta, train_meta, message='train meta')
    del train_examples, train_corpus

    # valid_meta = build_features(valid_examples, 'valid', 200, flags.valid_record_file, token2id)
    # save(flags.valid_eval_file, valid_evals, message='valid eval')
    # save(flags.valid_meta, valid_meta, message='valid_meta')
    # del valid_examples, valid_evals, valid_corpus

    test_meta = build_features(test_examples, 'test', config.max_len, flags.test_record_file, token2id,
                               flags.test_annotation)
    save(flags.test_meta, test_meta, message='test meta')
    del test_examples, test_corpus

    save(flags.shape_meta, {'max_len': config.max_len}, message='shape meta')
