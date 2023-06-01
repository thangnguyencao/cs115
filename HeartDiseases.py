import numpy as np
import pandas as pd
import random
import math

df1 = pd.read_csv('processed.cleveland.data', sep=",", header=None, prefix='')
df2 = pd.read_csv('reprocessed.hungarian.data', sep=" ", header=None, prefix='')
dataCleveland = df1.values
dataHungary = df2.values
dataset = np.vstack((dataCleveland,dataHungary))

def replace(dataset):
    for i in range(len(dataset)):
        if dataset[i][13] != 0:
            dataset[i][13] = 1
    return dataset

def Train_test_split(dataset):
    train = [list() for i in range(14)]
    for j in range(473):
        for i in range(14):
            train[i].append(dataset[j][i])
    test = [list() for i in range(14)]
    for j in range(473, 591):
        for i in range(14):
            test[i].append(dataset[j][i])
    return train, test


def Data_minmax_normalize(train, test):
    val_min = list()
    val_max = list()
    for i in range(13):
        val_min.append(min(train[i]))
        val_max.append(max(train[i]))
    for i in range(13):
        for j in range(len(train[i])):
            train[i][j] = (train[i][j] - val_min[i]) / (val_max[i] - val_min[i])
        for j in range(len(test[i])):
            test[i][j] = (test[i][j] - val_min[i]) / (val_max[i] - val_min[i])
    return train, test

def y_mu(kq, train, j):
    y = 0
    for i in range(13):
        y = y + kq[i] * train[i][j]
    y = y + kq[13]
    return y

def sigmoid(y):
    result = 1 / (1 + np.exp(-y))
    return result

def train_data(kq, train):
    ans = [0 for i in range(14)]
    for j in range(473):
        z = y_mu(kq, train, j)
        a = sigmoid(z)
        for i in range(13):
            ans[i] += (a - train[13][j]) * (train[i][j])
        ans[13] += (a - train[13][j])
    for i in range(14):
        ans[i] = ans[i] / len(train[0])
        kq[i] = kq[i] - 1 * ans[i]
    return kq

def Loss(kq, train):
    result = 0
    for j in range(473):
        a = sigmoid(y_mu(kq, train, j))
        result = result + (-train[13][j] * math.log(a, math.e)) - (1 - train[13][j]) * (math.log(1 - a, math.e))
    result = result / 473
    return result

def weighted_train_data(kq, train):
    ans = [0 for i in range(14)]
    for j in range(473):
        z = y_mu(kq, train, j)
        a = sigmoid(z)
        for i in range(13):
            ans[i] += (0.25 * a - 0.75 * train[13][j] + 0.5 * a * train[13][j]) * (train[i][j])
        ans[13] += (0.25 * a - 0.75 * train[13][j] + 0.5 * a * train[13][j])

    for i in range(14):
        ans[i] /= 473

    for i in range(14):
        kq[i] = kq[i] - 1 * ans[i]
    return kq

def weighted_Loss(kq, train):
    result = 0
    for j in range(473):
        a = sigmoid(y_mu(kq, train, j))
        result += (-0.75 * train[13][j] * math.log(a, math.e)) - 0.25 * (1 - train[13][j]) * (math.log(1 - a, math.e))
    result = result / 473
    return result

def confusion_matrix(kq, test):
    confusion_matrix = [[0, 0], [0, 0]]
    for j in range(len(test[0])):
        a = sigmoid(y_mu(kq, test, j))
        if a > 0.5:
            a = 1
        else:
            a = 0
        if a == test[13][j]:
            if a == 1:
                confusion_matrix[1][1] += 1
            else:
                confusion_matrix[1][0] += 1
        else:
            if a == 1:
                confusion_matrix[0][1] += 1
            else:
                confusion_matrix[0][0] += 1
    return confusion_matrix

def test_statistics(dataset):
    print("Test:", len(dataset[0]), end="\t")
    count_1 = dataset[13].count(1)
    count_0 = dataset[13].count(0)
    print("1:", count_1, end="\t")
    print("0:", count_0)
def train_statistics(dataset):
    print("Train:", len(dataset[0]), end="\t")
    count_1 = dataset[13].count(1)
    count_0 = dataset[13].count(0)
    print("1:", count_1, end="\t")
    print("0:", count_0)

def print_cfm(matrix):
    model = {'Actualy Positive(1)': pd.Series([matrix[1][1], matrix[0][0]],index=['Predict Positive(1)', 'Predict Negative(0)']),
             'Actualy Negative(0)': pd.Series([matrix[0][1], matrix[1][0]],index=['Predict Positive(1)', 'Predict Negative(0)'])}
    print(pd.DataFrame(model))

def accuracy(confusion_matrix):
    return (confusion_matrix[1][1] + confusion_matrix[1][0]) / (confusion_matrix[1][1] + confusion_matrix[0][0] + confusion_matrix[1][0] + confusion_matrix[0][1])

def precision(confusion_matrix):
    return confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])

def recall(confusion_matrix):
    return confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][0])

def f(confusion_matrix):
    return (precision(confusion_matrix) * recall(confusion_matrix)) / (precision(confusion_matrix) + recall(confusion_matrix)) * 2


dataset = replace(dataset)

train, test = Train_test_split(dataset)

train, test = Data_minmax_normalize(train, test)

train_statistics(train)
test_statistics(test)

kq = [random.random() for i in range(14)]
L = 2000

while L > 0.5:
    kq = train_data(kq, train)
    L = Loss(kq, train)
print("TN 1")
cfm = confusion_matrix(kq, test)
print_cfm(cfm)
print("Loss :", L)
print("Accuracy :", accuracy(cfm))
print("Precision :", precision(cfm))
print("Recall :", recall(cfm))
print("F :", f(cfm))
print("------------------------------------------------------------")
while L > 0.2:
    kq = weighted_train_data(kq, train)
    L = weighted_Loss(kq, train)
print("TN 2")
cfm = confusion_matrix(kq, test)
print_cfm(cfm)
print("Loss :", L)
print("Accuracy :", accuracy(cfm))
print("Precision :", precision(cfm))
print("Recall :", recall(cfm))
print("F :", f(cfm))


