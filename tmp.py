import os
from nltk.tokenize import word_tokenize
from sru import SRU

SPACE = ' '
data_path = os.getcwd()
valid_path = os.path.join(data_path, 'data/raw_data/altlex_dev.tsv')
test_path = os.path.join(data_path, 'data/raw_data/altlex_gold.tsv')
train_path = os.path.join(data_path, 'data/raw_data/altlex_train_bootstrapped.tsv')
raw_path = os.path.join(data_path, 'data/raw_data/altlex.tsv')
# dictionary = {}
# with open(data_path, 'r', encoding='utf8') as fh:
#     for line in fh:
#         line = line.strip().split(' ')
#         fredist = nltk.FreqDist(line)
#         for localkey in fredist.keys():
#             if localkey in dictionary.keys():
#                 dictionary[localkey] = dictionary[localkey] + fredist[localkey]
#             else:
#                 # 如果字典中不存在
#                 dictionary[localkey] = fredist[localkey]  # 将当前词频添加到字典中
#
#     frequency = sorted(dictionary.items(), key=lambda x: x[1])
#     print(len(frequency))


def process_train(path):
    sen_engs, sen_sims, seg_engs, seg_sims, labels = [], [], [], [], []
    idx = 1
    with open(path, 'r', encoding='utf8') as fh:
        for line in fh:
            line = line.strip().split('\t')
            labels.append(int(line[0]))
            del line[0]
            if len(line) != 6:
                print(idx)
            eng = word_tokenize(SPACE.join(line[:3]).strip())
            seg_engs.append([word_tokenize(seg) for seg in line[:3]])
            sim = word_tokenize(SPACE.join(line[3:]).strip())
            seg_sims.append([word_tokenize(seg) for seg in line[3:]])
            # sen_engs.append(eng)
            # sen_sims.append(sim)
            idx += 1
    fh.close()

    return seg_engs, seg_sims, labels


def process_test(path):
    sentences, labels = [], []
    seg_sentences = []
    i = 1
    with open(path, 'r', encoding='utf8') as fh:
        for line in fh:
            line = line.strip().split('\t')
            num = int(line[-1])
            labels.append(0 if num == 0 else 1)
            del line[-1]
            if len(line) < 3:
                print(i)
            sentences.append(word_tokenize(SPACE.join(line).strip()))
            seg_sentences.append([word_tokenize(seg) for seg in line])
            i += 1
    fh.close()

    return seg_sentences, labels


def seg_length(sentences):
    seg_len = []
    for sen in sentences:
        seg_len.append((len(sen[0]), len(sen[1]), len(sen[2])))
    return seg_len


def stat_length(sentences):
    max_pre, max_alt, max_cur = 0, 0, 0
    for sen in sentences:
        max_pre = max(max_pre, len(sen[0]))
        max_alt = max(max_alt, len(sen[1]))
        max_cur = max(max_cur, len(sen[2]))

    return max_pre, max_alt, max_cur


def stat_altlex(eng_sentences, sim_sentences, labels):
    c_alt, nc_alt =[], []
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


def gen_annotation(eng_length, sim_length, max_length, path, labels):
    with open(os.path.join(path, 'annotations.txt'), 'w', encoding='utf8') as f:
        for el, sl, label in zip(eng_length, sim_length, labels):
            pre, alt, cur = el
            if sum(el) > max_length:
                cur -= pre + alt + cur - max_length
            annos = 'O ' * pre
            annos += 'C ' if label == 1 else 'NC ' * alt
            annos += 'O ' * cur
            f.write(annos.strip() + '\n')
            pre, alt, cur = sl
            if sum(sl) > max_length:
                cur -= pre + alt + cur - max_length
            annos = 'O ' * pre
            annos += 'C ' if label == 1 else 'NC ' * alt
            annos += 'O ' * cur
            f.write(annos.strip() + '\n')
    f.close()


# engs, sims, labels = process_train(train_path)
# sens, labels = process_test(test_path)
# english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%',
#                         '"', '``', '-', '\'\'']
# seg_eng_filtered = [[[word.lower() for word in seg if word not in english_punctuations] for seg in eng] for eng in engs]
# seg_sim_filtered = [[[word.lower() for word in seg if word not in english_punctuations] for seg in sim] for sim in sims]
# seg_test_filtered = [[[word.lower() for word in seg if word not in english_punctuations] for seg in sen] for sen in sens]

# stat_altlex(seg_eng_filtered, seg_sim_filtered, labels)
# eng_len = seg_length(seg_eng_filtered)
# sim_len = seg_length(seg_sim_filtered)
# print(stat_length(seg_eng_filtered))
# print(stat_length(seg_sim_filtered))
# print(stat_length(seg_test_filtered))
# gen_annotation(eng_len, sim_len, 200, data_path, labels)

# anno_path = os.path.join(data_path, 'data/processed_data/test_annotations.txt')
# with open(anno_path, 'r', encoding='utf8') as fh:
#     line = fh.readline().strip().split(' ')
#     print(line)
#     print(list(map(int, line)))

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import utils as nn_utils

batch_size = 3
max_length = 4
hidden_size = 4
n_layers = 2

tensor_in = torch.LongTensor([[1, 2, 3, 1], [1, 0, 0, 0], [4, 2, 0, 0]])
tensor_in = Variable(tensor_in)  # [batch, seq, feature], [2, 3, 1]
seq_lengths = torch.IntTensor([4, 1, 2])  # list of integers holding information about the batch size at each sequence step

# pack it
# pack = nn_utils.rnn.pack_padded_sequence(tensor_in, seq_lengths, batch_first=True)

# initialize
emb = nn.Embedding(5, 5, padding_idx=0)
# rnn = nn.RNN(5, hidden_size, n_layers, batch_first=True, bidirectional=True)
rnn = SRU(5, hidden_size, n_layers, bidirectional=True)
h0 = Variable(torch.randn(2*n_layers, batch_size, hidden_size))

# forward
sorted_seq_lengths, indices = torch.sort(seq_lengths, dim=0, descending=True)
_, desorted_indices = torch.sort(indices, descending=False)
tensor_in = tensor_in[indices]
# tensor_in = tensor_in.index_select(0, Variable(idx_sort))
# seq_lengths = list(seq_lengths[idx_sort])
in_emb = emb(tensor_in)
in_emb = nn_utils.rnn.pack_padded_sequence(in_emb, sorted_seq_lengths, batch_first=True)
out, state = rnn(in_emb, h0)
state = state.view(n_layers, 2, batch_size, hidden_size)
forward_state, backward_state = state[-1][0], state[-1][1]
last_state = torch.cat([forward_state, backward_state], dim=1)
# unpack
unpacked, _ = nn_utils.rnn.pad_packed_sequence(out, batch_first=True)
unpacked = unpacked[desorted_indices]
print('unpacked', unpacked)
# print(out)

print('hello world')
