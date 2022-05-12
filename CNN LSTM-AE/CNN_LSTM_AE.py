import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def prepareData(train_file):
    for i in range(len(train_file)):
        print(pd.read_csv(train_file[i]))
    print("Data processed")

def createModel():
    #CNN LSTM model creation
    model = tf.keras.Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, )))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=adam)
    print("Model created!")

def startTrain():
    print("Model trained!")

def trainModel():
    train_file = ["KITCHEN_data_record.csv", "LIVINGROOM_data_record.csv", 
                 "CENTRALIZED_data_record.csv"]

    test_file = ["KITCHEN_data_record.csv", "LIVINGROOM_data_record.csv", 
                 "CENTRALIZED_data_record.csv"]

    output_file = ["KITCHEN_result", "LIVINGROOM_result"]

    prepareData(train_file)
    #createModel()
    #startTrain()

if __name__ == "__main__":
    trainModel()
