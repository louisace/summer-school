from sklearn.datasets import fetch_mldata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import math

mnist = fetch_mldata('MNIST-original', data_home='../dataSet')
image, label = mnist['data'],mnist['target']

train_img, test_img, train_label, test_label = train_test_split(image, label, test_size=1/7.0, random_state=0)

def bayes(train_img, train_label):
    num = train_img.shape[0]
    classNum = Counter(train_label)
    classP = np.array([classNum[i]/num for i in range(10)])

    classPci = np.empty((10, train_img.shape[1]))
    classpcj = np.empty((10, train_img.shape[1]))
    for i in range(10):
        classPci[i] = train_img[np.where(train_label==i)].sum(axis=0)
        # 拉普拉斯修正
        classpcj[i] = (classPci[i] + 1)/(classNum[i] + 2)
    print(classpcj.shape)
    return classP, classpcj

def pre(test_img, test_label, classP, classpcj):
    pre = np.empty((test_img.shape[0]))
    for i in range(test_img.shape[0]):
        prob = np.empty(10)
        for j in range(10):
            temp = sum([1e-5 if test_img[i][k] == 0 else math.log(classpcj[j][k]) for k in range(test_img.shape[1])])
            prob[j] = np.array(math.log(classP[j]) + temp)
        pre[i] = np.argmax(prob)
    return pre, (pre==test_label).sum()/test_label.shape[0]

if __name__ == '__main__':
    classP, classpcj = bayes(train_img, train_label)
    pre, acc = pre(test_img, test_label, classP, classpcj)
    print(acc)
