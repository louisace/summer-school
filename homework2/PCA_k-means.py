import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
import random

mnist = fetch_mldata('MNIST-original', data_home='../dataSet')
image, label = mnist['data'], mnist['target']
train_image, test_image, train_label, test_label = train_test_split(image, label, test_size=1/7.0, random_state=0)


def PCA(train_image, k):
    train_image1 =  train_image - np.mean(train_image, axis=0)
    train_cov = np.cov(train_image1)
    eigVals,eigVects = linalg.eig(mat(train_cov))
    index = np.argsort(eigVals)
    selectVec = np.matrix(eigVects.T[index[:k]])
    low_data = train_image1 * selectVec
    redata = (low_data * selectVec) + train_image1

    return low_data, redata

def dis(x, y):
    return np.sqrt(np.sum((x-y)**2))

def k_means(train_image, k, iters):
    m, n = train_image.shape
    # 第一列为簇，第二列为距离
    cluster = np.zeros((m ,2))
    center = np.zeros((k, n))
    for i in range(k):
        index = int(np.random.uniform(0, m))
        center[i, :] = train_image[index, :]
    for iter in range(iters):
        print(iter)
        for i in range(train_image.shape[0]):
            mindis = 99999999999
            minindex = -1
            for j in range(k):
                d = dis(center[j,:], train_image[i, :])
                if d < mindis:
                    mindis = d
                    minindex = j
            if cluster[i, 0] != minindex:
                cluster[i, 0] = minindex
                cluster[i, 1] = mindis

        for i in range(k):
            center_image = train_image[(cluster[:,0] == i)]
            center[i, :] = np.mean(center_image, axis=0)

    return center, cluster

if __name__ == '__main__':
    center, cluster = k_means(train_image, 10, 1)
    print(center.shape)
    print(test_image.shape)
    pre = np.empty((test_image.shape[0]))
    for i in range(test_image.shape[0]):
        mindis = 99999999
        minindex = -1
        for j in range(10):
            d = dis(center[j,:], test_image[i, :])
            if d < mindis:
                d = mindis
                minindex = j
        pre[i] = j
    acc = (pre == test_label).sum()/(test_image.shape[0])
    print(acc)

