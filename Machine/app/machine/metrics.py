import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def calculate_rmse(real, predict):
    real = np.array(real)
    predict = np.array(predict)
    RMSE = mean_squared_error(real, predict)**0.5
    #return round(RMSE, 4)
    return RMSE

def calculate_OR(real, predict):
    real = np.array(real)
    predict = np.array(predict)
    cnt = 0
    for i in range(len(real)):
        if predict[i] > 1.015 * real[i]:
            cnt += 1
    #print("CNT, LEN: ",cnt, len(real))
    return cnt / len(real)

def calculate_PR(real, predict):
    real = np.array(real)
    predict = np.array(predict)
    cnt = 0
    for i in range(len(real)):
        if predict[i] < 0.985 * real[i]:
            cnt += 1
    #print("CNT, LEN: ",cnt, len(real))
    return cnt / len(real)

def calculate_DA(real, predict):
    return 0.5

def calculate_mape(real, predict):
    return 0.0