from csv import reader
import math
import random
import pandas as pd
#import numpy as np

filename = "pima-indians-diabetes.data.csv"

def load_csv(filename):
    #dataset = list()
    with open(filename, newline='') as file:
        csv_reader = reader(file)
        dataset = []
        for row in csv_reader:
            for col in row:
                dataset.append([])
            break

        for row in csv_reader:
            for i in range(9):
                dataset[i].append(float(row[i]))

    return dataset
def normalize_dataset(train, test):
    val_max = []
    val_min = []
    for i in range(9):
        val_max.append(max(train[i]))
        val_min.append(min(train[i]))
    for i in range(8):
        for j in range(len(test[i])):
            test[i][j] = (test[i][j] - val_min[i]/val_max[i] - val_min[i])
    for i in range(8):
        for j in range(len(train[i])):
            train[i][j] = (train[i][j] - val_min[i]/val_max[i] - val_min[i])
    return test, train

def weighted_train_data(wb, xy):
    ans = [0 for i in range(9)]
    for i in range(len(xy[0])):
        z = z_val(wb, xy, i)
        a = sigmoid(z)
        for j in range(8):
            ans[j] += (0.2*a - 0.8*xy[8][i] + 0.6*a*xy[8][i])*(xy[j][i])
        ans[8] +=  (0.2*a - 0.8*xy[8][i] + 0.6*a*xy[8][i])
    for i in range(9):
        ans[i] /= len(xy[0])
    for i in range(9):
        wb[i] = wb[i] - 0.001 * ans[i]
    return wb
def train_data(wb, xy):
    ans = [0 for i in range(9)]
    for i in range(len(xy[0])):
        z = z_val(wb, xy, i)
        a = sigmoid(z)
        for j in range(8):
            ans[j] += (a - xy[8][i])*(xy[j][i])
        ans[8] += (a - xy[8][i])
    for i in range(9):
        ans[i] /= len(xy[0])
    for i in range(9):
        wb[i] = wb[i] - 0.001 * ans[i]
    return wb
#def Loss (wb, xy):
#    ans = 0
#    for i in range(len(xy[0])):
#        a = sigmoid(z_val(wb,xy,i))
#        ans += (-xy[8][i]*math.log(a,math.e) - (1 - xy[8][i])*math.log(1 - a, math.e))
#    ans /= len(xy[0])
#    return ans
#def weighted_loss(wb, xy):
#    ans = 0
#    for i in range(len(xy[0])):
#        a = sigmoid(z_val(wb, xy, i))
#        ans += (-0.8 * xy[8][i] * math.log(a, math.e) - 0.2*(1 - xy[8][i]) * math.log(1 - a, math.e))
#    ans /= len(xy[0])
#    return ans
def z_val(w_b, x_y, n): # z= sum(w*x) +b
    sum = 0
    for i in range(8):
        sum += w_b[i] * x_y[i][n]
    ans = sum + w_b[8]
    return ans
#def sigmoid(z_val):
#    return (1/(1+np.exp(-z_val)))
def sigmoid(z_val):
    val = 1/(1 + math.e**(-z_val))
    return val
def confusion_matrix(wb, xy):
    confusion_matrix = [[0,0], [0,0]]
    for i in range(len(xy[0])):
        a = sigmoid(z_val(wb, xy, i))
        if a > 0.5:
            a = 1
        else:
            a = 0
        if a == xy[8][i]:
            if a == 1:
                confusion_matrix[1][1] += 1 #True positive
            else:
                confusion_matrix[1][0] += 1 #True negative
        else:
            if a == 1:
                confusion_matrix[0][1] += 1 #False positive
            else:
                confusion_matrix[0][0] += 1 #False negative
    return confusion_matrix
def PrintArray(dataset):
    for i in range(250):
        for j in range(9):
            print(round(dataset[j][i],3), end = "\t")
        print()

def Test_and_Train(array0, array1):
    test = [[] for i in range(9)]
    train = [[] for i in range(9)]
    for j in range(200):
        for i in range(9):
            train[i].append(array0[i][j])
    for j in range(50):
        for i in range(9):
            train[i].append(array1[i][j])
    for j in range(200, 250):
        for i in range(9):
            test[i].append(array0[i][j])
            test[i].append(array1[i][j])
    return test, train
def SplitData(dataset):
    array1 = [[] for i in range(9)]
    array0 = [[] for i in range(9)]
    for j in range(len(dataset[0])):
        for i in range(9):
            if dataset[8][j] == 0 and len(array0[i]) <= 250:
                array0[i].append(dataset[i][j])
            if dataset[8][j] == 1 and len(array1[i]) <= 250:
                array1[i].append(dataset[i][j])
    return array0, array1
def accuracy(confusion_matrix):
    return (confusion_matrix[1][1] + confusion_matrix[0][0]) / (confusion_matrix[1][1] + confusion_matrix[0][0] + confusion_matrix[1][0] + confusion_matrix[0][1])
def precision(confusion_matrix):
    return (confusion_matrix[1][1])/(confusion_matrix[1][1] + confusion_matrix[0][1])
def recall(confusion_matrix):
    return (confusion_matrix[1][1])/(confusion_matrix[1][1] + confusion_matrix[1][0])
def print_confusion_matrix(matrix):
    ma_tb = {'Actually Positive(1)' : pd.Series([matrix[1][1],matrix[0][0]], index = ['Predict Positive(1)', 'Predict Negative(0)']),
             'Actually Negative(0)' : pd.Series([matrix[0][1],matrix[1][0]], index = ['Predict Positive(1)', 'Predict Negative(0)'])}
    print(pd.DataFrame(ma_tb))
def f_val(confusion_matrix):
    return 2*(precision(confusion_matrix) * recall(confusion_matrix))/(precision(confusion_matrix) + recall(confusion_matrix))
#-------------------------------------------------------------------------------------------------

dataset = load_csv(filename)
array0, array1 = SplitData(dataset)
test, train = Test_and_Train(array0, array1)
test, train = normalize_dataset(train, test)

kq = [random.random() for i in range(9)]
print("After train:", kq)

L = 1000
while L > 0.173:
    kq = weighted_train_data(kq, train)
  #  L = weighted_loss(kq, train)
  #  print("Loss :", L)
print("Before train:", kq)
print_confusion_matrix(confusion_matrix(kq, test))

print("accuracy :", accuracy(confusion_matrix(kq, test)))
print("precision :", precision(confusion_matrix(kq, test)))
print("recall :", recall(confusion_matrix(kq, test)))
print("F :", f_val(confusion_matrix(kq, test)))