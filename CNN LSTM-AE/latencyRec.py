from pandas._libs.tslibs.timestamps import Timestamp
from pandas.core.tools.datetimes import to_datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import datetime, copy
import tflite_runtime.interpreter as tflite
from datetime import datetime
from sklearn import preprocessing as prc
from sklearn.preprocessing import MinMaxScaler

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

def predictModel():
    dataDir = ".//"
    trainFile = ["KITCHEN_data_record.csv", "LIVINGROOM_data_record.csv", 
                 "CENTRALIZED_data_record.csv"]

    testFile = ["KITCHEN_test.csv", "LIVINGROOM_test.csv", 
                 "CENTRALIZED_test.csv"]

    outputFile = ["KITCHEN_result", "LIVINGROOM_result",
                 "CENTRALIZED_result"]

    all_features = []
    all_labels = []
    all_test_features = []
    all_test_labels = []

    step = 7

    for i in range(len(trainFile)):
        print("Preparing Federated {} with {} steps".format(trainFile[i], step))
        train_feature, train_label = prepareData(dataDir + trainFile[i], step)
        test_feature, test_label = prepareData(dataDir + testFile[i], step)
        #predictions = teacherModel.predict(train_feature, verbose=False)
        all_features.append(train_feature)
        all_labels.append(train_label)
        all_test_features.append(test_feature)
        all_test_labels.append(test_label)
        #all_predictions.append(predictions)
     
    #teacherModel.predict(all_features[0])

#     converter = tf.lite.TFLiteConverter.from_keras_model("Federated_LIVINGROOM_result_CNN-LSTM_7_steps_200_epoch_model")
#     converter.optimizations = []
#     converter.experimental_new_converter = True
#     converter.post_training_quantize=True
#     converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
#     liteModel = converter.convert()
#     open('liteLiving.tflite', 'wb').write(liteModel)

    #studentModel.predict(all_features[0])
    #kitchenModel.predict(all_features[1])
    #livingModel.predict(all_features[0])

    interpreter = tflite.Interpreter(model_path="liteTeacher.tflite")
    interpreter.allocate_tensors()
    all_inferences = []

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    for i in range(len(all_test_features[1])):
        x_tensor = np.expand_dims(all_test_features[1][i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], x_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        all_inferences.append(output_data[0][0])
    #print(rmse(all_predictions[1], all_test_labels[1]))
    print(rmse(all_inferences, all_test_labels[1]))

if __name__ == "__main__":
    predictModel()
