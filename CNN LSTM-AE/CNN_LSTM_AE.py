from types import DynamicClassAttribute
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def prepareData(trainFile, step):
    trainData = pd.read_csv(trainFile, parse_dates=['Time']).drop(['Type'], axis=1)
    print(trainData.describe())
    trainData = trainData.values
    print(trainData)
        

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
    dataDir = ".//"
    trainFile = ["KITCHEN_data_record.csv", "LIVINGROOM_data_record.csv", 
                 "CENTRALIZED_data_record.csv"]

    testFile = ["KITCHEN_data_record.csv", "LIVINGROOM_data_record.csv", 
                 "CENTRALIZED_data_record.csv"]

    outputFile = ["KITCHEN_result", "LIVINGROOM_result"]

    nSteps = [5, 7, 9]
    for i in range(len(trainFile)):
        for step in nSteps:
            prepareData(dataDir + trainFile[i], step)
 
    #createModel()
    #startTrain()

if __name__ == "__main__":
    trainModel()
