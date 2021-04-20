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
from csv import writer
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from .db_connector import save_parameters, save_best_model
from .metrics import calculate_rmse, calculate_mape, calculate_DA, calculate_OR, calculate_PR
from ..models import Parameters

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

file = pd.DataFrame(columns=['RMSE', 'hidden_layer_size', 'stack_size', 'train_window', 'learningrate', 'epoch'])
file.to_csv(os.getcwd()+'/app/machine/models/parameters.csv')

tickers = ['005930','000660','035420','005935','051910','207940','005380','006400','035720','068270','000270','005490','012330','051900','066570','028260','017670','105560','036570','034730','096770','055550','032830','003550','090430','015760','018260','009150','086790','003670']
hidden_sizes = [10, 20]
stack_sizes = [1, 2]
train_windows = [1, 3]
lrs = [0.01]
epochs = [30]
input_size = 1
output_size = 1


class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Net, self).__init__()
        self.hidden_layer_size = hidden_dim
        self.num_layer = layers
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=False)
        self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True)
        self.hidden_cell = (torch.zeros(self.num_layer,1,self.hidden_layer_size),
                            torch.zeros(self.num_layer,1,self.hidden_layer_size))
    def forward(self, x):
        lstm_out, self.hidden_cell = self.lstm(x.view(len(x) ,1, -1), self.hidden_cell)
        predictions = self.fc(lstm_out.view(len(x), -1))
        return predictions[-1]

def train():
    for ticker in tickers :
        print("TRAIN ", ticker, " START")
        ticker = str(ticker)
        ticker = ticker + '.ks'
        PATH = os.getcwd() + '/app/machine/models/' + ticker[0:-3] + '.pt'
        MinRMSE = float('inf')
        # 자동화 시켜야 함, date를 변수화
        # 제일 최근 데이터를 집어 넣는 전제
        # 휴장일 때는 학습 하지 않는다
        now = datetime.now()
        before_one_year = now - relativedelta(years=1)
        now = str(now).split(" ")[0]
        before_one_year = str(before_one_year).split(" ")[0]
        start_date= before_one_year
        end_date= now

        df = web.DataReader(ticker, data_source='yahoo', start=start_date, end=end_date)
        all_data = df['Close'].values.astype(float)

        test_data_size = 20

        train_data = all_data[:-test_data_size]
        test_data = all_data[-test_data_size:]

        #Data Normalization
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))
        train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

        for hidden_layer_size, stack_size, train_window, learningrate, epoch in itertools.product(hidden_sizes, stack_sizes, train_windows, lrs, epochs) :
            params = Parameters(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                pred_date=datetime.utcnow()+timedelta(hours=9),
                hidden_layer_size=hidden_layer_size,
                stack_size=stack_size,
                train_window=train_window,
                learning_rate=learningrate,
                epoch=epoch
            )
            print('')
            train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
            model = Net(input_size, hidden_layer_size, output_size, stack_size)
            loss_function = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)
            for i in range(epoch):
                for seq, labels in train_inout_seq:
                    optimizer.zero_grad()
                    model.hidden_cell = (torch.zeros(stack_size, 1, model.hidden_layer_size),
                            torch.zeros(stack_size, 1, model.hidden_layer_size))
                    y_pred = model(seq)

                    single_loss = loss_function(y_pred, labels)
                    single_loss.backward()
                    optimizer.step()

                if i%25 == 1:
                    print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

            print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

            fut_pred = 20

            test_inputs = train_data_normalized[-fut_pred:].tolist()

            model.eval()

            for i in range(fut_pred):
                seq = torch.FloatTensor(test_inputs[-train_window:])
                with torch.no_grad():
                    model.hidden = (torch.zeros(stack_size, 1, model.hidden_layer_size),
                                    torch.zeros(stack_size, 1, model.hidden_layer_size))
                    test_inputs.append(model(seq).item())

            actual_predictions = scaler.inverse_transform(np.array(test_inputs[fut_pred:] ).reshape(-1, 1))
            print(len(actual_predictions),len(df['Close'][-fut_pred:]))

            params.RMSE = calculate_rmse(actual_predictions,df['Close'][-fut_pred:])
            params.MAPE = calculate_mape(actual_predictions,df['Close'][-fut_pred:])
            params.DA = calculate_DA(actual_predictions,df['Close'][-fut_pred:])
            params.OR = calculate_OR(actual_predictions,df['Close'][-fut_pred:])
            params.PR = calculate_PR(actual_predictions,df['Close'][-fut_pred:])

            params_id = save_parameters(params)

            #for best model table
            if  MinRMSE > params.RMSE :
                MinRMSE = params.RMSE
                torch.save({
                    'hidden_layer_size' : hidden_layer_size,
                    'stack_size' : stack_size,
                    'train_window' : train_window,
                    'learning_rate' : learningrate,
                    'epoch': epoch,
                    'rmse' : params.RMSE,
                    'train_data' : train_data_normalized[-fut_pred:].tolist(),
                    'scaler' : scaler,
                    'model': model,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, PATH)
                save_best_model(ticker, PATH, params_id)

            #for parameter table
            with open(os.getcwd()+'/app/machine/models/parameters.csv', 'a') as f_object:

                writer_object = writer(f_object)
                writer_object.writerow([ticker, params.RMSE, hidden_layer_size, stack_size, train_window, learningrate, epoch])
                f_object.close()

if __name__ == '__main__':
    # scheduler.every.do(train)
    train()
