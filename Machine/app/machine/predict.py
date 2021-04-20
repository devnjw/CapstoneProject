import os
import torch
import torch.nn as nn

import pandas_datareader as web
import seaborn as sns
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from csv import writer

from .db_connector import save_result

class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Net, self).__init__()
        self.hidden_layer_size = hidden_dim
        self.num_layer = layers
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=False)
        self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True)
        self.hidden_cell = (torch.zeros(self.num_layer,self.output_dim,self.hidden_layer_size),
                            torch.zeros(self.num_layer,self.output_dim,self.hidden_layer_size))
    def forward(self, x):
        lstm_out, self.hidden_cell = self.lstm(x.view(len(x) ,1, -1), self.hidden_cell)
        predictions = self.fc(lstm_out.view(len(x), -1))
        return predictions[-1]

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    test_inputs = checkpoint['train_data']
    train_window = checkpoint['train_window']
    scaler = checkpoint['scaler']
    model = checkpoint['model']
    print(checkpoint['rmse'])
    print(checkpoint['epoch'])
    print(checkpoint['learning_rate'])
    model.load_state_dict(checkpoint['model_state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model, test_inputs, train_window, scaler

tickers = ['005930','000660','035420','005935','051910','207940','005380','006400','035720','068270','000270','005490','012330','051900','066570','028260','017670','105560','036570','034730','096770','055550','032830','003550','090430','015760','018260','009150','086790','003670']
def predict_loop():
    for ticker in tickers:
        predict(ticker)

def predict(ticker):
    print("PREDICT ", ticker , " START")
    model, test_inputs, train_window, scaler = load_checkpoint(os.getcwd()+'/app/machine/models/'+ ticker +'.pt')

    #variable for no.of days you want to predict
    fut_pred = 20

    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden = (torch.zeros(model.num_layer, 1, model.hidden_layer_size),
                            torch.zeros(model.num_layer, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item())

    actual_predictions = scaler.inverse_transform(np.array(test_inputs[fut_pred:] ).reshape(-1, 1))

    #for results table
    print(actual_predictions)
    save_result(actual_predictions, ticker)

if __name__ == '__main__':
    predict()
