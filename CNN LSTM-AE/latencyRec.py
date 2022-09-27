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
import datetime, copy
from datetime import datetime
from sklearn import preprocessing as prc
from sklearn.preprocessing import MinMaxScaler
from Centralized_CNN_LSTM import prepareData, createStudentModel, rmse

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
    all_predictions = []

    allKitchenRmse = []
    allLivingRmse = []
    teacherModel = keras.models.load_model("CENTRALIZED_result_CNN-LSTM_7_steps_86_epoch_model")
    studentModel = keras.models.load_model("_student_CENTRALIZED_result_CNN-LSTM_7_steps_121_epoch_model")
    kitchenModel = keras.models.load_model("Federated_KITCHEN_result_CNN-LSTM_7_steps_200_epoch_model")
    livingModel = keras.models.load_model("Federated_LIVINGROOM_result_CNN-LSTM_7_steps_200_epoch_model")

    step = 7


    for i in range(len(trainFile)):
        print("Preparing Federated {} with {} steps".format(trainFile[i], step))
        train_feature, train_label = prepareData(dataDir + trainFile[i], step)
        test_feature, test_label = prepareData(dataDir + testFile[i], step)
        predictions = teacherModel.predict(train_feature, verbose=False)
        all_features.append(train_feature)
        all_labels.append(train_label)
        all_test_features.append(test_feature)
        all_test_labels.append(test_label)
        all_predictions.append(predictions)
     
    #teacherModel.predict(all_features[0])

    converter = tf.lite.TFLiteConverter.from_keras_model(livingModel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    liteModel = converter.convert()
    open('liteLiving.tflite', 'wb').write(liteModel)

    #studentModel.predict(all_features[0])
    #kitchenModel.predict(all_features[1])
    #livingModel.predict(all_features[0])

    interpreter = tf.lite.Interpreter(model_path="liteLiving.tflite")
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
    print(rmse(all_predictions[1], all_test_labels[1]))
    print(rmse(all_inferences, all_test_labels[1]))

if __name__ == "__main__":
    predictModel()
