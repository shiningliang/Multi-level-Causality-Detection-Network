import fasttext
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

path = '../data/raw_data'
train = 'train_training.csv'
boot = 'train_boot.csv'
dev = 'dev.csv'

# lines = open(os.path.join(path, train), 'r', encoding='ISO-8859-1').readlines()
# fw = open(os.path.join(path, 'train_training.txt'), 'w', encoding='utf-8')
# for line in lines:
#     line = line.strip().split(',')
#     if line[0] == 'labels':
#         continue
#     label = '__label__' + line[0]
#     fw.write(label + ' ' + line[1] + '\n')
#
# fw.close()

# fast = os.path.join(path, 'cc.en.300.bin')
# model = fasttext.load_model(fast)

for ep in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    model = fasttext.train_supervised(input='../data/raw_data/train_boot.txt', lr=1.0, epoch=ep, wordNgrams=2)
    print(model.test('../data/raw_data/dev.txt'))

print('hello world')
