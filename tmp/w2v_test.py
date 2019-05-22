import gensim
from gensim.scripts import glove2word2vec
import os
import shutil
from sys import platform
import pickle

path = 'data/processed_data'

google = os.path.join(path, 'GoogleNews-vectors-negative300.bin')
model = gensim.models.KeyedVectors.load_word2vec_format(google, binary=True)
# glove6B = os.path.join(path, 'glove.840B.bin')
# model = gensim.models.KeyedVectors.load_word2vec_format(glove6B, binary=True)
# model.save_word2vec_format(os.path.join(path, 'glove.840B.bin'), binary=True)

word_weights = {}
for word in model.vocab:
    word_weights[word] = model[word]
with open(os.path.join(path, 'google.news.pkl'), 'wb') as file:
    pickle.dump(word_weights, file)


# 计算行数，就是单词数
def getFileLineNums(filename):
    f = open(filename, 'r')
    count = 0
    for line in f:
        count += 1
    return count


# Linux或者Windows下打开词向量文件，在开始增加一行
def prepend_line(infile, outfile, line):
    with open(infile, 'r') as old:
        with open(outfile, 'w') as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)


def prepend_slow(infile, outfile, line):
    with open(infile, 'r') as fin:
        with open(outfile, 'w') as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)


def load(filename):
    num_lines = getFileLineNums(filename)
    gensim_file = 'glove_model.txt'
    gensim_first_line = "{} {}".format(num_lines, 300)
    # Prepends the line.
    if platform == "linux" or platform == "linux2":
        prepend_line(filename, gensim_file, gensim_first_line)
    else:
        prepend_slow(filename, gensim_file, gensim_first_line)

    model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file)
    print(model['word'])


# glove_6B = os.path.join(path, 'glove.6B.300d.txt')
# glove_6B_out = os.path.join(path, 'glove.6B.txt')
# load(glove_6B)
# print(glove2word2vec.get_glove_info(glove_6B))
# glove2word2vec.glove2word2vec(glove_6B, glove_6B_out)

# glove_840B = os.path.join(path, 'glove.840B.300d.txt')
# glove_840B_out = os.path.join(path, 'glove.840B.txt')
# load(glove_840B)
# print(glove2word2vec.get_glove_info(glove_840B))
# glove2word2vec.glove2word2vec(glove_840B, glove_840B_out)
