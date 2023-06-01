import numpy as np
import pandas as pd

def load_csv(filename):
    df = pd.read_csv(filename, header=None).astype('float64')
    return df

def normalize_dataframe(df):
    for i in range(df.columns[-1]):
        df[i] = (df[i]-df[i].min()) / (df[i].max()-df[i].min())
    return df

def split_df(df):
    df0 = df.loc[df.iloc[:, -1] == 0].head(250)
    df1 = df.loc[df.iloc[:, -1] == 1].head(250)
    return pd.concat([df0, df1], axis=0).reset_index(drop=True)

def split_train_data_x(df, ratio):
    x0_train = df.iloc[:, :-1].head(int(250 * ratio))
    x1_train = df.iloc[250:, :-1].head(int(250 * ratio))
    return pd.concat([x0_train, x1_train], axis=0).reset_index(drop=True)

def split_test_data_x(df, ratio):
    x0_test = df.iloc[:250, :-1].tail(int(250 * ratio))
    x1_test = df.iloc[:500, :-1].tail(int(250 * ratio))
    return pd.concat([x0_test, x1_test], axis=0).reset_index(drop=True)

def split_train_data_y(df, ratio):
    y0_train = df.iloc[:, -1:].head(int(250 * ratio))
    y1_train = df.iloc[250:, -1:].head(int(250 * ratio))
    return pd.concat([y0_train, y1_train], axis=0).reset_index(drop=True)

def split_test_data_y(df, ratio):
    y0_test = df.iloc[:250, -1:].tail(int(250 * ratio))
    y1_test = df.iloc[:500, -1:].tail(int(250 * ratio))
    return pd.concat([y0_test, y1_test], axis=0).reset_index(drop=True)

def weightInitialization(n_features):
    w = np.zeros((1,n_features))
    b = 0
    return w, b


def sigmoid_activation(result):
    return (1/(1+np.exp(-result)))

def cost(sigmoid, y_train):
    return ((-1/y_train.shape[0]) * (np.sum( (y_train.T * np.log(sigmoid)) + ((1-y_train.T) * np.log(1-sigmoid)) )))

def gradient_descent(sigmoid, x_train, y_train):
    w_new = (1/x_train.shape[0]) * np.dot(x_train.T, (sigmoid - y_train.T).T)
    b_new = (1/x_train.shape[0]) * np.sum(sigmoid - y_train.T)
    return w_new, b_new

def model_predict(w, b, df, train_ratio, learning_rate):
    # train
    x_train = split_train_data_x(df, train_ratio).to_numpy()
    y_train = split_train_data_y(df, train_ratio).to_numpy()
    
    # for i in range(iterations):
    costs = 1
    i = 0
    while (costs > 0.5):
        sigmoid = sigmoid_activation(np.dot(w, x_train.T) + b)
        costs = cost(sigmoid, y_train)
        w_new, b_new = gradient_descent(sigmoid, x_train, y_train)
        w = w - (learning_rate * w_new.T)
        b = b - (learning_rate * b_new)
        print(i)
        i += 1
        #print(w_new, b_new)
        if costs < 1e-10:
            break
    
    # test
    #x_test = split_test_data_x(df, round((1 - train_ratio), 3))
    #y_test = split_test_data_y(df, round((1 - train_ratio), 3))
    
    return costs

def predict(final_pred, m):
    y_pred = np.zeros((1,m))
    for i in range (final_pred.shape[1]):
        if final_pred[0][i] > 0.5:
            y_pred[0][i] =1
            return y_pred

#=============================================================================

filename = 'pima-indians-diabetes.data.csv'
df = load_csv(filename)
df = normalize_dataframe(df)
df = split_df(df)
w, b = weightInitialization(df.columns[-1])
costs = model_predict(w, b, df, 0.8, 0.01)

print("cost = " + str(costs))
#=============================================================================

