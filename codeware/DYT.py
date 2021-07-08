import os
from sklearn import metrics
from sklearn.metrics import roc_auc_score,roc_curve, auc
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import xlsxwriter
import xlrd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.io as sio
from sklearn import svm



#3-mers编码
base = ['A','G','C','U']
def codenX(seq_list):  #特征编码
    # 字典创建
    coden_dict = {}
    m = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                coden_dict[base[i] + base[j] + base[k]] = m
                m = m + 1
    vectors = np.zeros((87, 1))#64+23
    seq = seq_list[0]  # 序列
    for i in range(len(seq)-2):
        vectors[coden_dict[seq[i:i + 3].replace('T', 'U')]][0] += 1
    vectors[coden_dict["AGG"]][0] -= 1
    vectors[coden_dict["GGA"]][0] -= 1
    vectors[coden_dict["GAG"]][0] -= 1
    for i in range(23):
        vectors[64 + i][0] = seq_list[i + 1]
    vectors = normorlize(vectors)  # 归一化
    return vectors.tolist()
# def codenX(seq_list):  #特征编码
#     # 字典创建
#     vectors = np.zeros((23, 1))
#     seq = seq_list[0]  # 序列
#     for i in range(23):
#         vectors[i][0] = seq_list[i + 1]
#     vectors = normorlize(vectors)  # 归一化
#     return vectors.tolist()


def codenY(score):#标签编码
    if float(score) == -1:
        vector = [0, -1]  # 稳定
    else:
        vector = [1, 1]  # 不稳定


    return vector
def codenH(seq_list):  #热图矩阵编码
    vectors = np.zeros((22, 1))
    for i in range(len(seq_list) - 2):
        vectors[i][0] = seq_list[i + 1]
    return vectors.tolist()

def normorlize(my_matrix):# 归一化数组
    scaler = MinMaxScaler()
    scaler.fit(my_matrix)
    scaler.data_max_
    my_matrix_normorlize = scaler.transform(my_matrix)
    return my_matrix_normorlize

def DYT():
    # 数据准备
    path='../dataset/DYT.xlsx'
    dataX = []
    dataY = []
    data_xsls = xlrd.open_workbook(path)  # 打开此地址下的excel文档
    sheet_name = data_xsls.sheets()[0]  # 进入第一张表
    count_nrows = sheet_name.nrows  # 获取总行数
    for i in range(1, count_nrows):
        line_value = sheet_name.row_values(i)
        dataX.append(codenX(line_value))
        dataY.append(codenY(line_value[-1]))
    indexes = np.random.choice(len(dataY), len(dataY), replace=False)
    dataX = np.array(dataX)[indexes][:, :, -1]
    dataY = np.array(dataY)[indexes]
    train_X, test_X, train_Y, test_Y= train_test_split(dataX, dataY,test_size=0.3)

    # 一分类
    model = svm.OneClassSVM(nu=0.05, kernel='rbf', gamma=0.1)
    model.fit(train_X, train_Y[:, -1])

    y_pred = model.predict(test_X)
    ACC = np.mean(y_pred== test_Y[:, 1])
    P = metrics.precision_score(test_Y[:, 1], y_pred, average='binary')  # Precision准确率
    R = metrics.recall_score(test_Y[:, 1], y_pred, average='binary')  # Recall召回率
    F1 = metrics.f1_score(test_Y[:, 1], y_pred, average='binary')
    print(ACC, P, R, F1)


if __name__ == "__main__":
    DYT()
