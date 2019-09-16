import os
import pickle as pkl
from tqdm import tqdm
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import ujson as json
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


def check_null(sen):
    flag = False
    if len(sen) == 3:
        # if len(sen[0]) > 0:
        #     pre = sen[0]
        # else:
        #     pre = ['<NULL>']
        #     flag = True
        # if len(sen[1]) > 0:
        #     mid = sen[1]
        # else:
        #     mid = ['<NULL>']
        #     flag = True
        # if len(sen[2]) > 0:
        #     cur = sen[2]
        # else:
        #     cur = ['<NULL>']
        #     flag = True
        pre = sen[0] if len(sen[0]) > 0 else ['<NULL>']
        mid = sen[1] if len(sen[1]) > 0 else ['<NULL>']
        cur = sen[2] if len(sen[2]) > 0 else ['<NULL>']
    else:
        pre = sen[0] if len(sen[0]) > 0 else ['<NULL>']
        mid = ['<NULL>']
        cur = ['<NULL>']
        flag = True

    return pre, mid, cur, flag


def preprocess_train(file_path, file_name, data_type, is_build=False):
    print("Generating {} examples...".format(data_type))
    examples = []
    engs, sims = [], []
    seg_engs, seg_sims, labels = [], [], []
    data_path = os.path.join(file_path, file_name)
    lines = open(data_path, 'r', encoding='ISO-8859-1').readlines()
    for line in lines:
        line = line.strip().split('\t')
        labels.append(int(line[0]))
        del line[0]
        if is_build:
            engs.append(word_tokenize(SPACE.join(line[:3]).strip()))
            sims.append(word_tokenize(SPACE.join(line[3:]).strip()))
        seg_engs.append([word_tokenize(seg) for seg in line[:3]])
        seg_sims.append([word_tokenize(seg) for seg in line[3:]])

    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%',
                            '"', '``', '-', '\'\'']
    if is_build:
        eng_filtered = [[word.lower() for word in document if word not in english_punctuations] for document in engs]
        sim_filtered = [[word.lower() for word in document if word not in english_punctuations] for document in sims]
    seg_eng_filtered = [[[word.lower() for word in seg if word not in english_punctuations] for seg in eng] for eng
                        in seg_engs]
    seg_sim_filtered = [[[word.lower() for word in seg if word not in english_punctuations] for seg in sim] for sim
                        in seg_sims]

    total = 0
    seq_len = []
    for label, eng, sim in zip(labels, seg_eng_filtered, seg_sim_filtered):
        total += 1
        pre, mid, cur, flag = check_null(eng)
        if flag:
            print(total)
        examples.append({'eid': total,
                         'tokens': pre + mid + cur,
                         'tokens_pre': pre,
                         'tokens_alt': mid,
                         'tokens_cur': cur,
                         'cau_label': label})
        seq_len.append(len(pre + mid + cur))

        total += 1
        pre, mid, cur, flag = check_null(sim)
        if flag:
            print(total)
        examples.append({'eid': total,
                         'tokens': pre + mid + cur,
                         'tokens_pre': pre,
                         'tokens_alt': mid,
                         'tokens_cur': cur,
                         'cau_label': label})
        seq_len.append(len(pre + mid + cur))

    if is_build:
        sentences = []
        for eng_tokens, sim_tokens in zip(eng_filtered, sim_filtered):
            sentences.append(SPACE.join(eng_tokens))
            sentences.append(SPACE.join(sim_tokens))
    else:
        sentences = []
    np.random.shuffle(examples)
    stat_length(seq_len)
    return examples, sentences, (seg_eng_filtered, seg_sim_filtered), labels


def preprocess_test(file_path, file_name, data_type, is_build=False):
    print("Generating {} examples...".format(data_type))
    examples = []
    sentences, segments, labels = [], [], []
    data_path = os.path.join(file_path, file_name)
    lines = open(data_path, 'r', encoding='ISO-8859-1').readlines()
    for line in lines:
        line = line.strip().split('\t')
        num = int(line[-1])
        del line[-1]
        labels.append(0 if num == 0 else 1)
        sentences.append(word_tokenize(SPACE.join(line).strip()))
        if len(line) == 3:
            segments.append([word_tokenize(seg) for seg in line])
        else:
            segments.append([['<NULL>'], word_tokenize(line[0]), word_tokenize(line[1])])

    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%',
                            '"', '``', '-', '\'\'']
    if is_build:
        sen_filtered = [[word.lower() for word in sentence if word not in english_punctuations] for sentence in
                        sentences]
    seg_filtered = [[[word.lower() for word in seg if word not in english_punctuations] for seg in eng] for eng in
                    segments]
    total = 0
    seq_len = []
    for label, seg in zip(labels, seg_filtered):
        total += 1
        pre, mid, cur, flag = check_null(seg)
        if flag:
            print(total)
        examples.append({'eid': total,
                         'tokens': pre + mid + cur,
                         'tokens_pre': pre,
                         'tokens_alt': mid,
                         'tokens_cur': cur,
                         'cau_label': label})
        seq_len.append(len(pre + mid + cur))
    # print('Get {} total examples'.format(total))
    # print('Get {} causal examples'.format(causal))
    # print('Get {} non-causal examples'.format(non_causal))
    if is_build:
        sentences = [SPACE.join(tokens) for tokens in sen_filtered]
    else:
        sentences = []
    stat_length(seq_len)
    return examples, sentences, seg_filtered, labels


def preprocess_transfer(file_path, file_name, data_type, is_build=False):
    print("Generating {} examples...".format(data_type))
    examples = []
    data_path = os.path.join(file_path, file_name)

    total = 0
    with open(data_path, 'rb') as f:
        data_set = json.load(f)
    f.close()
    for label, sample in zip(data_set['label'], data_set['sample']):
        total += 1
        pre, mid, cur, flag = check_null(sample)
        if flag:
            print(total)
        examples.append({'eid': total,
                         'tokens': pre + mid + cur,
                         'tokens_pre': pre,
                         'tokens_alt': mid,
                         'tokens_cur': cur,
                         'cau_label': label})
    return examples


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

    # token2id = {'<NULL>': 0, '<OOV>': 1, '<LEARN>': 2}
    token2id = {'<NULL>': 0, '<OOV>': 1}
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, 'rb') as fin:
            trained_embeddings = pkl.load(fin)
        fin.close()
        embedding_dict = set(trained_embeddings)
        print('Num of tokens in corpus {}'.format(len(corpus_dict)))
        filtered_tokens = corpus_dict.intersection(embedding_dict)  # common
        oov_tokens = corpus_dict.difference(filtered_tokens)
        combined_tokens = []
        for token in oov_tokens:
            if len(token.split('-')) > 1:
                combined_tokens.append(token)
        combined_tokens = set(combined_tokens)
        # oov_tokens = oov_tokens.difference(combined_tokens)
        # token2id = {'<NULL>': 0, '<OOV>': 1}
        # embedding_mat = np.zeros([len(corpus_dict) + 2, vec_size])
        embedding_mat = np.zeros([len(filtered_tokens) + len(token2id), vec_size])
        for token in filtered_tokens:
            token2id[token] = len(token2id)
            embedding_mat[token2id[token]] = trained_embeddings[token]

        combined = 0
        for tokens in combined_tokens:
            sub_tokens = tokens.split('-')
            token_vec = np.zeros([vec_size])
            in_emb = 0
            for t in sub_tokens:
                if t in filtered_tokens:
                    token_vec += trained_embeddings[t]
                    in_emb += 1
            if in_emb > 0:
                combined += 1
                token2id[tokens] = len(token2id)
                embedding_mat = np.row_stack((embedding_mat, token_vec / in_emb))
        scale = 3.0 / max(1.0, (len(corpus_dict) + vec_size) / 2.0)
        embedding_mat[1] = np.random.uniform(-scale, scale, vec_size)
        print('Filtered_tokens: {} Combined_tokens: {} OOV_tokens: {}'.format(len(filtered_tokens),
                                                                              combined,
                                                                              len(oov_tokens)))
    else:
        embedding_mat = np.random.uniform(-0.25, 0.25, (len(corpus_dict) + len(token2id), vec_size))
        embedding_mat[0] = np.zeros(vec_size)
        embedding_mat[1] = np.zeros(vec_size)
        for token in corpus_dict:
            token2id[token] = len(token2id)
    # id2token = dict([val, key] for key, val in token2id.items())
    id2token = dict(zip(token2id.values(), token2id.keys()))
    # print(len(token2id), len(id2token), len(embedding_mat))
    return embedding_mat, token2id, id2token


def gen_embedding(data_type, corpus_dict, emb_file=None, vec_size=None):
    print("Generating {} embedding...".format(data_type))

    token2id = {'<NULL>': 0, '<OOV>': 1}
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
        # token2id = {'<NULL>': 0, '<OOV>': 1}
        # embedding_mat = np.zeros([len(corpus_dict) + 2, vec_size])
        embedding_mat = np.zeros([len(filtered_tokens) + 2, vec_size])
        for token in filtered_tokens:
            token2id[token] = len(token2id)
            embedding_mat[token2id[token]] = trained_embeddings[token]

        combined = 0
        for tokens in combined_tokens:
            sub_tokens = tokens.split('-')
            token_vec = np.zeros([vec_size])
            in_emb = 0
            for t in sub_tokens:
                if t in filtered_tokens:
                    token_vec += trained_embeddings[t]
                    in_emb += 1
            if in_emb > 0:
                combined += 1
                token2id[tokens] = len(token2id)
                embedding_mat = np.row_stack((embedding_mat, token_vec / in_emb))
        scale = 3.0 / max(1.0, (len(corpus_dict) + vec_size) / 2.0)
        embedding_mat[1] = np.random.uniform(-scale, scale, vec_size)
        # for token in oov_tokens:
        #     token2id[token] = len(token2id)
        #     embedding_mat[token2id[token]] = np.random.uniform(-scale, scale, vec_size)\
        print('Filtered_tokens: {} Combined_tokens: {} OOV_tokens: {}'.format(len(filtered_tokens),
                                                                              combined,
                                                                              len(oov_tokens)))
    else:
        embedding_mat = np.random.uniform(-0.25, 0.25, (len(corpus_dict) + 2, vec_size))
        embedding_mat[0] = np.zeros(vec_size)
        embedding_mat[1] = np.zeros(vec_size)
        for token in corpus_dict:
            token2id[token] = len(token2id)
    id2token = dict([val, key] for key, val in token2id.items())
    # print(len(token2id), len(id2token), len(embedding_mat))
    return embedding_mat, token2id, id2token


def seg_length(sentences):
    seg_len = []
    for sen in sentences:
        seg_len.append((len(sen[0]), len(sen[1]), len(sen[2])))
    return seg_len


def gen_annotation(segs, max_length, filename, labels, data_type):
    max_length = max_length['full']
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
    # fh = open(annotation_file, 'r', encoding='utf8')
    for sentence in tqdm(sentences):
        total += 1
        tokens = np.zeros([max_len['full']], dtype=np.int32)
        tokens_pre = np.zeros([max_len['pre']], dtype=np.int32)
        tokens_alt = np.zeros([max_len['alt']], dtype=np.int32)
        tokens_cur = np.zeros([max_len['cur']], dtype=np.int32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2id:
                    return word2id[each]
            return 1

        seq_len = min(len(sentence['tokens']), max_len['full'])
        pre_len = min(len(sentence['tokens_pre']), max_len['pre'])
        alt_len = min(len(sentence['tokens_alt']), max_len['alt'])
        cur_len = min(len(sentence['tokens_cur']), max_len['cur'])
        for i in range(seq_len):
            tokens[i] = _get_word(sentence['tokens'][i])
        for i in range(pre_len):
            tokens_pre[i] = _get_word(sentence['tokens_pre'][i])
        for i in range(alt_len):
            tokens_alt[i] = _get_word(sentence['tokens_alt'][i])
        for i in range(cur_len):
            tokens_cur[i] = _get_word(sentence['tokens_cur'][i])
        samples.append({'id': sentence['eid'],
                        'tokens': tokens,
                        'tokens_pre': tokens_pre,
                        'tokens_alt': tokens_alt,
                        'tokens_cur': tokens_cur,
                        'length': seq_len,
                        'cau_label': sentence['cau_label']})
    # fh.close()
    with open(out_file, 'wb') as fo:
        pkl.dump(samples, fo)
    fo.close()
    print('Build {} instances of features in total'.format(total))
    meta['total'] = total
    return meta


def run_prepare(config):
    train_examples, train_corpus, train_seg, train_labels = preprocess_train(config.raw_dir, config.train_file,
                                                                             'train', config.build)
    transfer_examples1 = preprocess_transfer(config.raw_dir, config.transfer_file1, 'transfer')
    transfer_examples2 = preprocess_transfer(config.raw_dir, config.transfer_file2, 'transfer')
    valid_examples, valid_corpus, valid_seg, valid_labels = preprocess_test(config.raw_dir, config.valid_file,
                                                                            'valid', config.build)
    test_examples, test_corpus, test_seg, test_labels = preprocess_test(config.raw_dir, config.test_file,
                                                                        'test', config.build)

    if config.build:
        # types = ['train', 'valid', 'test']
        # labels = [train_labels, valid_labels, test_labels]
        # segs = [train_seg, valid_seg, test_seg]
        # for t, s, l in zip(types, segs, labels):
        #     gen_annotation(s, config.max_len, os.path.join(config.processed_dir, t + '_annotations.txt'), l, t)
        save(config.corpus_file, train_corpus, 'corpus')
        corpus_dict = build_dict(config.corpus_file)
        token_emb_mat, token2id, id2token = get_embedding('word', corpus_dict, config.w2v_file, config.n_emb)
        save(config.token_emb_file, token_emb_mat, message='embeddings')
        save(config.token2id_file, token2id, message='token to index')
        save(config.id2token_file, id2token, message='index to token')
    else:
        with open(config.token2id_file, 'r') as fh:
            token2id = json.load(fh)

    transfer_meta1 = build_features(transfer_examples1, 'transfer', config.max_len, config.transfer_record_file1,
                                    token2id)
    save(config.transfer_meta1, transfer_meta1, message='transfer meta')
    del transfer_examples1
    transfer_meta2 = build_features(transfer_examples2, 'transfer', config.max_len, config.transfer_record_file2,
                                    token2id)
    save(config.transfer_meta2, transfer_meta2, message='transfer meta')
    del transfer_examples2

    train_meta = build_features(train_examples, 'train', config.max_len, config.train_record_file, token2id,
                                config.train_annotation)
    save(config.train_meta, train_meta, message='train meta')
    del train_examples, train_corpus

    valid_meta = build_features(valid_examples, 'valid', config.max_len, config.valid_record_file, token2id)
    save(config.valid_meta, valid_meta, message='valid meta')
    del valid_examples, valid_corpus

    test_meta = build_features(test_examples, 'test', config.max_len, config.test_record_file, token2id,
                               config.test_annotation)
    save(config.test_meta, test_meta, message='test meta')
    del test_examples, test_corpus

    save(config.shape_meta, {'max_len': config.max_len}, message='shape meta')
