import os
import nltk

data_path = os.getcwd()
data_path = os.path.join(data_path, 'data/processed_data/corpus.txt')

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

    frequency = sorted(dictionary.items(), key=lambda x: x[1])
    print(len(frequency))
