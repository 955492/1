import keras
import numpy as np
from sklearn.model_selection import train_test_split
import csv
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.metrics import accuracy_score

#Load data from csv document
def load_Data():
    csv_file = csv.reader(open('Data_.csv','r'))
    Y = list()
    X = list()
    for line in csv_file:
        Y.append(float(line[1]))
        line[2] = line[2].replace('[', '')
        line[2] = line[2].replace(']','')
        line[2] = line[2].replace(', ','')
        tmp1 = list()
        tmp2 = list()
        for i in range(len(line[2])):
            tmp2.append(line[2][i])
            if (i + 1) % 5 == 0:
                tmp1.append(tmp2)
                tmp2 = list()
        X.append(tmp1)
    return X, Y

#Split the dataset into trainset and testset
def split_data(train_x, train_y):
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size = 0.3, random_state = 42)
    return X_train, X_test, y_train, y_test

def build_model(look_back, batch_size):
    """
    The function builds a keras Sequential model
    :param look_back: number of previous time steps as int
    :param batch_size: batch_size as int, defaults to 1
    :return: keras Sequential model
    """
    model = Sequential()
    model.add(LSTM(64,
                   activation='relu',
                   batch_input_shape=(batch_size, look_back, 1),
                   stateful=True,
                   return_sequences=False))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


if __name__ == '__main__':
    X, Y = load_Data()
    X_train, X_test, Y_train, Y_test = split_data(X, Y)
    #print(X_train, X_test, y_train, y_test)
    look_back = int(len(X) * 0.20)
    train_size = int(len(X) * 0.70)
    batch_size = 50
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    model = build_model(look_back, batch_size=batch_size)
    model.fit(Y_test, Y_train, batch_size=batch_size, epochs=1, verbose=0, shuffle=False)
    #model.fit


