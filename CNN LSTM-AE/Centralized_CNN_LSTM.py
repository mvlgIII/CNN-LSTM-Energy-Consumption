from pandas._libs.tslibs.timestamps import Timestamp
from pandas.core.tools.datetimes import to_datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.keras import optimizers, layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, Flatten, TimeDistributed
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Reshape, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import datetime
from datetime import datetime
from sklearn import preprocessing as prc
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

#class CustomCallback(keras.callbacks.Callback):
#    def on_epoch_end(self, epoch, logs=None):
#        self.model.save(dataDir + outputFile[2] + '_CNN-LSTM_' + str(step) + "_steps_{}_epoch_model".format(epoch))
        
#Data preprocessing - Formats the data file to a usable format for the model
def prepareData(trainFile, step):
    trainData = pd.read_csv(trainFile).drop(['Type'], axis=1)
    #print(trainData.head())

    #Time interval between previous and current sample

    #Date Time Format
    #trainData['Time'] = trainData['Time'].apply(lambda x: datetime.strptime(trainData['Time'][x]).timestamp(), axis=1)
    trainData['Time'] = pd.to_datetime(trainData['Time'])
    #for i in range(len(trainData)):
    #    trainData['Time'][i] = trainData['Time'][i].timestamp()
    #    print(trainData['Time'][i])
    
    #Time Interval Format
    timeIntervals = []
    timeInterval = trainData['Time'][0] - trainData['Time'][0]
    timeInterval = timeInterval.total_seconds()
    timeIntervals.append(timeInterval)
    for i in range(len(trainData)):
        if i > 0:
            timeInterval = trainData['Time'][i] - trainData['Time'][i-1]
            timeInterval = timeInterval.total_seconds()
            timeIntervals.append(timeInterval)
    #print(timeIntervals)
    trainData['TimeInterval'] = timeIntervals
    trainData['TimeInterval'] = trainData['TimeInterval'].to_numpy().astype(np.float64)
    trainData = trainData.drop(['Time'], axis=1)

    scaler = MinMaxScaler()
    trainData = scaler.fit_transform(trainData)
    trainData = pd.DataFrame(trainData)
    #print(trainData.head())
    trainData = trainData.values

    feature = []
    label = []
    for i in range(len(trainData)):
        end = i + step
        if end > len(trainData) - 1:
            break
        seq_x, seq_y = trainData[i:end, ], trainData[end, 0] * trainData[end, 1]
        feature.append(seq_x)
        label.append(seq_y)
        
    feature, label = np.array(feature).astype(np.float64), np.array(label).astype(np.float64)
    #print("Shape of feature: {}".format(feature.shape))
    return feature, label

#Model creation - Assembles and compiles the whole network model to be used
def createTeacherModel(shape):
    #CNN LSTM model creation
    tf.random.set_seed(1)
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(shape[1], shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(300, activation='sigmoid', return_sequences=True)))
    model.add(Dense(1, activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adadelta', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.summary()
    return model

def createStudentModel(shape):
    model = Sequential()
    model.add(Conv1D(filters=12, kernel_size=2, activation='relu', input_shape=(shape[1], shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(37, activation='sigmoid', return_sequences=True)))
    model.add(Dense(1, activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer='adadelta',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.summary()
    return model

#Training - Fits the model into the data and begins the training while recording metric values
def startTrain(model, train_feature, train_label, validation_feature, validation_label, metricFile, epoch):
    dataInfo = model.fit(train_feature, train_label, epochs=1, batch_size=32, #batch size multiple of 2^x, early stopping00
                         validation_data=(validation_feature, validation_label),
                         callbacks=[EarlyStopping(monitor='loss', patience=3)])
    metric = open(metricFile, 'a')
    for i in range(len(dataInfo.history['loss'])):
        metric.write('{}, {}, {}, {}, {}\n'.format(epoch + 1, dataInfo.history['loss'][i],
                                                   dataInfo.history['val_loss'][i],
                                                   dataInfo.history['root_mean_squared_error'][i],
                                                   dataInfo.history['val_root_mean_squared_error'][i]))
    metric.close()
    return model

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


#Function that contains all the process to train the model
def trainModel():
    dataDir = ".//"
    trainFile = ["KITCHEN_data_record.csv", "LIVINGROOM_data_record.csv", 
                 "CENTRALIZED_data_record.csv"]

    testFile = ["KITCHEN_test.csv", "LIVINGROOM_test.csv", 
                 "CENTRALIZED_test.csv"]

    outputFile = ["KITCHEN_result", "LIVINGROOM_result",
                 "CENTRALIZED_result"]
    
    centralPredictData = ".//CENTRALIZED_predict.csv"
    testPredict = ".//data_recordNEW-CENTRALIZED.csv"
    model = 0
    all_test_features = []
    all_test_labels = []

    nSteps = [5, 7]
    #Centralized learning approach:
    for step in nSteps:
        #All data preparation
        for i in range(len(trainFile)):
            print("Preparing data for Centralized {}_{}".format(trainFile[i], step))
            train_feature, train_label = prepareData(dataDir + trainFile[i], step)
            test_feature, test_label = prepareData(dataDir + testFile[i], step)
            if i < 2:
                all_test_features.append(test_feature)
                all_test_labels.append(test_label)

        #Teacher model data preprocessing
        print("Shape of validation: {}".format(all_test_features[0].shape))
        model = createTeacherModel(train_feature.shape)
        metricFile = dataDir + outputFile[2] + "_CNN-LSTM_" + str(step) + "_steps.csv"
        print("Training " + metricFile)
        fle = open(metricFile, 'w')
        fle.write('epoch, trainloss, validationloss, trainRMSE, validationRMSE\n')
        fle.close()
        allKitchenRmse = []
        allLivingRmse = []

        #Teacher model training
        for epoch in range(200):
            print("Training epoch round {}".format(epoch+1))
            model = startTrain(model, train_feature, train_label, test_feature, test_label, metricFile, epoch)
            kitchenPrediction = model.predict(all_test_features[0], verbose=False)
            livingPrediction = model.predict(all_test_features[1], verbose=False)
            kitchenRmse = rmse(kitchenPrediction, all_test_labels[0])
            livingRmse = rmse(livingPrediction, all_test_labels[1])
            allKitchenRmse.append(kitchenRmse)
            allLivingRmse.append(livingRmse)
            model.save(dataDir + outputFile[2] + '_CNN-LSTM_' + str(step) + "_steps_{}_epoch_model".format(epoch + 1))
        results = pd.read_csv(metricFile)
        results['kitchenRMSE'] = allKitchenRmse
        results['livingRMSE'] = allLivingRmse
        results.to_csv(metricFile, mode='w', header=True, index=False)

        x = range(len(allKitchenRmse))
        plt.figure()
        plt.plot(x, allKitchenRmse)
        plt.plot(x, allLivingRmse)
        plt.legend(['Kitchen', 'Living Room'])
        plt.title("Centralized Teacher Model Validation")
        #plt.show()

        

        #Student model data preprocessing
        predict_feature, predict_label = prepareData(centralPredictData, step) #Features for student
        #predictions = model.predict(predict_feature[:14637], verbose=False) # Labels for student

        predictions = model.predict(predict_feature, verbose=False)
        print("Feature: {} --> Predictions: {}".format(len(predict_feature), len(predictions)))
        val = input()

        studentModel = createStudentModel(train_feature.shape)
        studentMetricFile = dataDir + "_student_" + outputFile[2] + "_CNN-LSTM_" + str(step) + "_steps.csv"
        fle = open(studentMetricFile, 'w')
        fle.write('epoch, trainloss, validationloss, trainRMSE, validationRMSE\n')
        fle.close()
        print("Training student model of {}".format(trainFile[2]))

        allKitchenRmse.clear()
        allLivingRmse.clear()
        for epoch in range(200):
            print("Training epoch round {}".format(epoch+1))
            studentModel = startTrain(studentModel, predict_feature, predictions, test_feature, test_label, studentMetricFile, epoch)
            kitchenPrediction = studentModel.predict(all_test_features[0], verbose=False)
            livingPrediction = studentModel.predict(all_test_features[1], verbose=False)
            kitchenRmse = rmse(kitchenPrediction, all_test_labels[0])
            livingRmse = rmse(livingPrediction, all_test_labels[1])
            allKitchenRmse.append(kitchenRmse)
            allLivingRmse.append(livingRmse)
            studentModel.save(dataDir + "_student_" + outputFile[2] + '_CNN-LSTM_' + str(step) + "_steps_{}_epoch_model".format(epoch+1))

        results = pd.read_csv(studentMetricFile)
        results['kitchenRMSE'] = allKitchenRmse
        results['livingRMSE'] = allLivingRmse
        results.to_csv(studentMetricFile, mode='w', header=True, index=False)

        x = range(len(allKitchenRmse))
        plt.figure()
        plt.plot(x, allKitchenRmse)
        plt.plot(x, allLivingRmse)
        plt.legend(['Kitchen', 'Living Room'])
        plt.title("Centralized Student Model Validation")
        #plt.show()

        

        #For debugging/verification
        #testPredict_feature, testPredict_label = prepareData(testPredict, step)
        #studentPredictions = studentModel.predict(testPredict_feature, verbose=False)

        #print("Teacher model prediction values: {}".format(predictions))
        #print("Student model prediction values: {}".format(studentPredictions))

        

    #Main function
if __name__ == "__main__":
    trainModel()