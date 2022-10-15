import time
import board
import busio
import math
import threading
import keyboard
import csv
import datetime
import queue
import traceback
import concurrent.futures
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from Adafruit_IO import *

username = 'jorrelbrandonz'
accKey = 'aio_YvNa08cYcfAaWbCSc4zTzaomMzkk'
aio = Client(username, accKey)
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c)
A0 = AnalogIn(ads, ADS.P0)
A1 = AnalogIn(ads, ADS.P3)
samples = 200
stepValue = 66 #mV
connType = 0
header_text = ['Date', 'Voltage', 'Current', 'Connection Type']
endApp = False


def getOffset():
    # Getting offset for current calculation
    value = []
    print('Get offset voltage')
    for idx in range(10):
        count = int(0)
        maxCurrent = 0
        dataCurrent = []
        
        for count in range(samples):
            dataCurrent.insert(count, A0.voltage)
            if dataCurrent[count] > maxCurrent:
                maxCurrent = dataCurrent[count]
        
        value.append(maxCurrent)
        
    sum = 0
    for val in value:
        sum += val
    offset = sum/len(value)
    offset = round (offset, 3)
    print('Offset: {}'.format(offset))
    return offset


def runMonitor(offset, classif, monitorData):
    global endApp
    global A0, A1
    count = int(0)
    maxVoltage = 0
    maxCurrent = 0
    classification = 0
    while not endApp:
        dataVoltage = []
        dataCurrent = []
        Vrms = 0
        Irms = 0
        peakVoltage = 0
        peakCurrent = 0
        samples = 200
        for count in range(samples):
            try:
                dataVoltage.insert(count, A1.voltage)
                dataCurrent.insert(count, A0.voltage)
                    
                if dataVoltage[count] > maxVoltage:
                    maxVoltage = dataVoltage[count]
                if dataCurrent[count] > maxCurrent:
                    maxCurrent = dataCurrent[count]
                    maxCurrent = round(maxCurrent, 3)
            except OSError:
                i2c = busio.I2C(board.SCL, board.SDA)
                ads = ADS.ADS1115(i2c)
                A0 = AnalogIn(ads, ADS.P0)
                A1 = AnalogIn(ads, ADS.P1)
                print("Reset")
            except IndexError:
                pass
            
        if(maxCurrent < 0):
            maxCurrent = 0
        peakVoltage = float(maxVoltage * 80.6)
        peakCurrent = float((maxCurrent - offset) / 0.066)
        Vrms = peakVoltage/math.sqrt(2)
        Irms = peakCurrent/math.sqrt(2)
        #print("Current: {:>5.3f}".format(maxCurrent))
        maxVoltage = 0
        maxCurrent = 0
        #print("{:>5.3f}\t{:>5.3f}\t".format(Vrms, peakCurrent))
        currentTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            classification = classif.get(block=False)
        except queue.Empty:
            pass
        
        data = [currentTime, round(peakVoltage, 3), round(peakCurrent, 3), classification]
        monitorData.put(data, block=False)
        time.sleep(10)

def saveFile(monitorData):
    global endApp
    while not endApp:
        try:
            data = monitorData.get(block=True, timeout=None)
            print(data)
            with open('data_record.csv', 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(data)
                csv_file.close()
        except queue.Empty:
            pass

if __name__ == '__main__':
    
    offset = getOffset()
    print("{:>5}\t{:>5}".format('voltage', 'current'))
    classif = queue.Queue(maxsize=1)
    monitorData = queue.Queue(maxsize=1)
    t1 = threading.Thread(target=runMonitor,args=(offset, classif, monitorData))
    t2 = threading.Thread(target=saveFile, args=(monitorData,))
    try:
        t2.start()
        t1.start()
        while not endApp:
            classification = input("Input Classification: ")
            classif.put(classification, block=False)
    except KeyboardInterrupt:
        #traceback.print_exc()
        endApp = True
        
    t2.join()
    t1.join()
    
    #monitorData.insert(3, connType)
   # print(monitorData)
#         try:
#             with open('data_record.csv', 'w') as csv_file:
#                 writer = csv.DictWriter(csv_file, fieldnames=header_text)
#                 writer.writeheader()
#         except:
#                 print('Error')                     
    #try:
     #   with open('data_record.csv', 'a', newline='') as csv_file:
#             csv_writer = csv.writer(csv_file)
#             csv_writer.writerow(monitorData)
#             csv_file.close()
#     except csv.Error:
#         print('An error has occured with the writing, please try again')
    
    #time.sleep(50)        
        
            
            
    
