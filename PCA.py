from sklearn.decomposition import KernelPCA,PCA
import numpy as np
from util import LoadData_pickle
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from BaseSVDD import BaseSVDD
from sklearn import metrics
import pickle

# 创建PCA模型，指定主成分数量（这里为3）
kpca = KernelPCA(n_components=300)

# 拟合模型并进行数据转换
for class_num in range(8):
    data_train = LoadData_pickle(path='./data/AUV/', name=f'C{class_num}_train')
    data_train_PCA = kpca.fit_transform(data_train)
    file_train = open('./data/AUV/PCA/C' + str(class_num) + '_train_PCA.pkl', 'wb')
    pickle.dump(data_train_PCA, file_train)
    file_train.close()

    data_test = LoadData_pickle(path='./data/AUV/', name=f'C{class_num}_test')
    data_test_PCA = kpca.fit_transform(data_test)
    file_test = open('./data/AUV/PCA/C' + str(class_num) + '_test_PCA.pkl', 'wb')
    pickle.dump(data_test_PCA, file_test)
    file_test.close()

