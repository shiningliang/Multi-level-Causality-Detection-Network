import os
from nltk.tokenize import word_tokenize

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
    i = 1
    with open(path, 'r', encoding='utf8') as fh:
        for line in fh:
            line = line.strip().split('\t')
            num = int(line[-1])
            labels.append(0 if num == 0 else 1)
            del line[0]
            if len(line) < 3:
                print(i)
            tmp = word_tokenize(SPACE.join(line).strip())
            sentences.append(tmp)
            i += 1
    fh.close()

    return sentences, labels


def seg_max(sentences):
    max_b, max_l, max_a = 0, 0, 0
    for sen in sentences:
        max_b = max(max_b, len(sen[0]))
        max_l = max(max_l, len(sen[1]))
        max_a = max(max_a, len(sen[2]))

    return max_b, max_l, max_a


engs, sims, labels = process_train(train_path)
# sens, labels = process_test(test_path)
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%',
                        '"', '``', '-', '\'\'']

seg_eng_filtered = [[[word.lower() for word in seg if word not in english_punctuations] for seg in eng] for eng in engs]
seg_sim_filtered = [[[word.lower() for word in seg if word not in english_punctuations] for seg in sim] for sim in sims]

emax_b, emax_l, emax_a = seg_max(seg_eng_filtered)
smax_b, smax_l, smax_a = seg_max(seg_sim_filtered)

print('hello world')
