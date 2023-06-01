from csv import reader

import numpy as np

learning_rate = 0.01
w = np.random.rand(11)
b = np.random.rand(1)
w_new = [0] * 11
def csv_reading(filename):
    dataset = list()
    with open(filename, 'r' ) as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column]=float(row[column].strip())
def str_to_float(dataset):
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        min_value = min(col_values)
        max_value = max(col_values)
        minmax.append([min_value, max_value])
    return minmax
def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i]=(row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
filename = 'winequality-white.csv'
dataset = csv_reading(filename)
str_to_float(dataset)
minmax=dataset_minmax(dataset)
normalize(dataset, minmax)
#print(dataset)
def new_w (w, b, subscript, row, column, dataset):
    ans = 0
    for i in range (len(row)):
        temp = 0
        for j in range(len(column) - 1):
            temp += w[j] * dataset[i][j]
        ans += (temp + b - dataset[i][11] * dataset[i][subscript])
    return ans * (1/row)
def new_b (w, b, row, column, dataset):
    ans = 0
    for i in range(len(row)):
        temp = 0
        for j in range(len(column) - 1):
            temp += w[j] * dataset[i][j]
        ans += (temp + b - dataset[i][11])
    return ans * (1/row)
def Loss(w, b, row, column):
    ans = 0
    for i in range(len(row)):
        temp = 0
        for j in range(len(column) - 1):
            temp += w[j] * dataset[i][j]
        ans += ((temp + b - dataset[i][11]) ** 2)
    return ans * (1/row)
def MSE(w, b, startRow, endRow, totalRow, column, dataset):
    ans = 0
    for i in range(startRow, endRow):
        temp = 0
        for j in range(column - 1):
            temp +=w[j] * dataset[i][j]
        ans += ((temp + b - dataset[i][11]) ** 2 )
    return ans * (1/totalRow)

Train_row = int (row * 80/100)
MSE_row = (row - Train_row)
dataTrain = dataset[0:Train_row]
for i in range(10):
    print(i)
    for subscript in range(len(w)):
        w_new[subscript] = w[subscript] - learning_rate*new_w(w, b, subscript, Train_row, column, dataset)
        b_new = b - learning_rate * new_b(w, b, Train_row, column, dataset)
        L = Loss(w, b, Train_row, column, dataset)
    if (L < 1e-3):
        break
    w = w_new
    b = b_new

for i in range(len(w)):
    print(w[i])
print(b)


print("--------")
mse = MSE(w, b, Train_row, row, MSE_row, column, dataset)
print(mse)
