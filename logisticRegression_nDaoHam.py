from csv import reader
import numpy as np
import math

#############################
w1 = np.random.rand(1)          ##
w2 = np.random.rand(1)         ##
w3 = np.random.rand(1)  ##
w4 = np.random.rand(1)         ##
w5 = np.random.rand(1)          ##
w6 = np.random.rand(1)             ##
w7 = np.random.rand(1)       ##
w8 = np.random.rand(1)       ##
b = np.random.rand(1)     ##
lr = 0.01                   ##
#############################

def load_csv (filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def str_column_to_float (dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
def str_to_float(dataset):
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)

def dataset_minmax(dataset):
    minmax = list()
    for i in range (len(dataset[0])):
        col_val = [row[i] for row in dataset]
        val_min = min(col_val)
        val_max = max(col_val)
        minmax.append([val_min, val_max])
    return minmax

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range (len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


#####################################################################################################
filename = "pima-indians-diabetes.data.csv"       ##
dataset = load_csv(filename)                                                                       ##
str_to_float(dataset)                                                                              ##
minmax = dataset_minmax(dataset)                                                                   ##
normalize_dataset(dataset, minmax)                                                                 ##
#####################################################################################################

list_0 = []
list_1 = []
for i in range(len(dataset)):
    if int(dataset[i][8]) == 1:
        list_1.append(dataset[i])
    else:
        list_0.append(dataset[i])

training_list = list_0[:200] + list_1[:200] + list_0[200:250] + list_1[200:250]

def dao_ham_w1 (w1, w2, w3, w4, w5, w6, w7, w8, b):
    answer_w1 = 0
    for i in range(400, 500):
        tmp = 1/100 * ((1 / (1 + math.exp(-(w1*training_list[i][0] + w2*training_list[i][1] + w3*training_list[i][2] + w4*training_list[i][3] + w5*training_list[i][4] + w6*training_list[i][5] + w7*training_list[i][6] + w8*training_list[i][7] + b)))) - training_list[i][8]) * training_list[i][0]
        answer_w1 += tmp
    return answer_w1

def dao_ham_w2 (w1, w2, w3, w4, w5, w6, w7, w8, b):
    answer_w2 = 0
    for i in range(400, 500):
        tmp = 1/100 * ((1 / (1 + math.exp(-(w1*training_list[i][0] + w2*training_list[i][1] + w3*training_list[i][2] + w4*training_list[i][3] + w5*training_list[i][4] + w6*training_list[i][5] + w7*training_list[i][6] + w8*training_list[i][7] + b)))) - training_list[i][8]) * training_list[i][1]
        answer_w2 += tmp
    return answer_w2

def dao_ham_w3 (w1, w2, w3, w4, w5, w6, w7, w8, b):
    answer_w3= 0
    for i in range(400, 500):
        tmp = 1/100 * ((1 / (1 + math.exp(-(w1*training_list[i][0] + w2*training_list[i][1] + w3*training_list[i][2] + w4*training_list[i][3] + w5*training_list[i][4] + w6*training_list[i][5] + w7*training_list[i][6] + w8*training_list[i][7] + b)))) - training_list[i][8]) * training_list[i][2]
        answer_w3 += tmp
    return answer_w3

def dao_ham_w4 (w1, w2, w3, w4, w5, w6, w7, w8, b):
    answer_w4 = 0
    for i in range(400, 500):
        tmp = 1/100 * ((1 / (1 + math.exp(-(w1*training_list[i][0] + w2*training_list[i][1] + w3*training_list[i][2] + w4*training_list[i][3] + w5*training_list[i][4] + w6*training_list[i][5] + w7*training_list[i][6] + w8*training_list[i][7] + b)))) - training_list[i][8]) * training_list[i][3]
        answer_w4 += tmp
    return answer_w4

def dao_ham_w5 (w1, w2, w3, w4, w5, w6, w7, w8, b):
    answer_w5 = 0
    for i in range(400,500):
        tmp = 1/100 * ((1 / (1 + math.exp(-(w1*training_list[i][0] + w2*training_list[i][1] + w3*training_list[i][2] + w4*training_list[i][3] + w5*training_list[i][4] + w6*training_list[i][5] + w7*training_list[i][6] + w8*training_list[i][7] + b)))) - training_list[i][8]) * training_list[i][4]
        answer_w5 += tmp
    return answer_w5

def dao_ham_w6 (w1, w2, w3, w4, w5, w6, w7, w8, b):
    answer_w6 = 0
    for i in range(400,500):
        tmp = 1/100 * ((1 / (1 + math.exp(-(w1*training_list[i][0] + w2*training_list[i][1] + w3*training_list[i][2] + w4*training_list[i][3] + w5*training_list[i][4] + w6*training_list[i][5] + w7*training_list[i][6] + w8*training_list[i][7] + b)))) - training_list[i][8]) * training_list[i][5]
        answer_w6 += tmp
    return answer_w6

def dao_ham_w7 (w1, w2, w3, w4, w5, w6, w7, w8, b):
    answer_w7 = 0
    for i in range(400,500):
        tmp = 1/100 * ((1 / (1 + math.exp(-(w1*training_list[i][0] + w2*training_list[i][1] + w3*training_list[i][2] + w4*training_list[i][3] + w5*training_list[i][4] + w6*training_list[i][5] + w7*training_list[i][6] + w8*training_list[i][7] + b)))) - training_list[i][8]) * training_list[i][6]
        answer_w7 += tmp
    return answer_w7

def dao_ham_w8 (w1, w2, w3, w4, w5, w6, w7, w8, b):
    answer_w8 = 0
    for i in range(400,500):
        tmp = 1/100 * ((1 / (1 + math.exp(-(w1*training_list[i][0] + w2*training_list[i][1] + w3*training_list[i][2] + w4*training_list[i][3] + w5*training_list[i][4] + w6*training_list[i][5] + w7*training_list[i][6] + w8*training_list[i][7] + b)))) - training_list[i][8]) * training_list[i][7]
        answer_w8 += tmp
    return answer_w8

def dao_ham_b (w1, w2, w3, w4, w5, w6, w7, w8, b):
    answer_b = 0
    for i in range(400,500):
        tmp = 1/100 * ((1 / (1 + math.exp(-(w1*training_list[i][0] + w2*training_list[i][1] + w3*training_list[i][2] + w4*training_list[i][3] + w5*training_list[i][4] + w6*training_list[i][5] + w7*training_list[i][6] + w8*training_list[i][7] + b)))) - training_list[i][8])
        answer_b += tmp
    return answer_b

def Loss (w1, w2, w3, w4, w5, w6, w7, w8, b) :
    answer_c = 0
    for i in range(400, 500) :
        tmp =1/100 * ((-training_list[i][8])*math.log((1 / (1 + math.exp(-(w1*training_list[i][0] + w2*training_list[i][1] + w3*training_list[i][2] + w4*training_list[i][3] + w5*training_list[i][4] + w6*training_list[i][5] + w7*training_list[i][6] + w8*training_list[i][7] + b))))) - (1 - training_list[i][8]) * math.log((1 / (1 + math.exp(-(w1*training_list[i][0] + w2*training_list[i][1] + w3*training_list[i][2] + w4*training_list[i][3] + w5*training_list[i][4] + w6*training_list[i][5] + w7*training_list[i][6] + w8*training_list[i][7] + b))))))
        answer_c += tmp
    return answer_c


for i in range(10000):
    w1_new = w1 - lr*dao_ham_w1(w1, w2, w3, w4, w5, w6, w7, w8, b)
    w2_new = w2 - lr*dao_ham_w2(w1, w2, w3, w4, w5, w6, w7, w8, b)
    w3_new = w3 - lr*dao_ham_w3(w1, w2, w3, w4, w5, w6, w7, w8, b)
    w4_new = w4 - lr*dao_ham_w4(w1, w2, w3, w4, w5, w6, w7, w8, b)
    w5_new = w5 - lr*dao_ham_w5(w1, w2, w3, w4, w5, w6, w7, w8, b)
    w6_new = w6 - lr*dao_ham_w6(w1, w2, w3, w4, w5, w6, w7, w8, b)
    w7_new = w7 - lr*dao_ham_w7(w1, w2, w3, w4, w5, w6, w7, w8, b)
    w8_new = w8 - lr*dao_ham_w8(w1, w2, w3, w4, w5, w6, w7, w8, b)
    b_new = b - lr*dao_ham_b(w1, w2, w3, w4, w5, w6, w7, w8, b)
    L = Loss (w1, w2, w3, w4, w5, w6, w7, w8, b)
    if L < 1e-10 :
        break
    w1 = w1_new
    w2 = w2_new
    w3 = w3_new
    w4 = w4_new
    w5 = w5_new
    w6 = w6_new
    w7 = w7_new
    w8 = w8_new
    b = b_new

    count = 0
for i in range(400,500):
    #tmp1 = 0
    tmp = (1 / (1 + math.exp(-(w1*training_list[i][0] + w2*training_list[i][1] + w3*training_list[i][2] + w4*training_list[i][3] + w5*training_list[i][4] + w6*training_list[i][5] + w7*training_list[i][6] + w8*training_list[i][7] + b))))
    if tmp >= 0.5 :
        tmp = 1
    else :
        tmp = 0
    if (tmp == int(training_list[i][8])):
        count += 1
    print(tmp, training_list[i][8])


pre = count / 100
print("predict = " + str(pre))
'''print(w1, w2, w3, w4, w5, w6, w7, w8)
print(b)'''
print("Loss = " + str(L))
