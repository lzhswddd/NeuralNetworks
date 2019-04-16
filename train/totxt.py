import os

def get_files(filename):  # 提取文件夹下文件名、目录
    class_train = []
    label_train = []
    word = 'ABCDEFGHJKLMNPRSTUVWXYZ'
    word = list(word)
    word_dirt = {}
    for i in range(len(word)):
        word_dirt[word[i]] = i
    for train_class in os.listdir(filename):
        for pic in os.listdir(filename + '/' + train_class):
            class_train.append(train_class + '/' + pic)
            out = [0 for i in range(23)]
            out[word_dirt[train_class]] = 1
            label_train.append(out)
    return class_train, label_train


path, label = get_files('./images')

with open('data.txt', 'w') as file:
    for i in range(len(path)):
        file.write(path[i] + '|')
        for l in label[i]:
            file.write(str(l)+' ')
        file.write('\n')
