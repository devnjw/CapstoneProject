from flask import render_template, make_response, request, flash, session, g, jsonify, Response
from flask_restful import Resource
from werkzeug.security import generate_password_hash, check_password_hash

import pandas as pd
import numpy as np
import pandas_datareader as web
import json

from app.models import User, Results, TickerInfo
from .. import db
import plotly.express as px

from datetime import datetime, timedelta

class Predict(Resource):
    def post(self):
        input_stock = request.form['ticker']
        Ticker = TickerInfo.query.filter_by(ticker=input_stock).first()
        if not Ticker:
            Ticker = TickerInfo.query.filter_by(kor_name=input_stock).first()
        if not Ticker:
            Ticker = TickerInfo.query.filter_by(eng_name=input_stock).first()
        if not Ticker:
            flash('Database에 존재하지 않는 종목입니다.')
            return make_response(render_template('stock.html'))
        ticker = Ticker.ticker

        print("Ticker: ", ticker)

        #now = datetime.datetime.now()
        #print(now) # 2015-04-19 12:11:32.669083

        yesterday = datetime.today() - timedelta(1)
        past365 = datetime.today() - timedelta(30)

        yesterday = yesterday.strftime('%Y-%m-%d')
        past365 = past365.strftime('%Y-%m-%d')
        
        values, dates = get_results(ticker)
        #print("Values: ", values)
        #print("Dates: ", dates)
        
        df = web.DataReader(f'{ticker}.KS', data_source='yahoo', start=f'{past365}', end=f'{yesterday}')
        df=df.reset_index()
  
        fig = px.line(df, x='Date', y='Close', title='Time Series with Rangeslider')
        dic = {}
        fig.update_xaxes(rangeslider_visible=True)
        #fig.show()
        

        results = Results.query.filter_by(ticker=ticker).order_by(Results.pred_date.desc()).limit(41).all()

        pred01 = []
        pred05 = []
        pred20 = []
        real = []
        for result in results:
            pred01.append(result.d01)
            pred05.append(result.d05)
            pred20.append(result.d20)
            real.append(result.d00)

        DA01 = calculate_DA(real, pred01, 1) # Should be placed before shortenning list length to 20
        DA05 = calculate_DA(real, pred05, 5)
        DA20 = calculate_DA(real, pred20, 20)

        pred01 = pred01[1:21]
        pred05 = pred05[5:25]
        pred20 = pred20[20:40]
        real = real[0:len(pred20)]

        
        RMSE01 =  calculate_rmse(real, pred01)
        RMSE05 =  calculate_rmse(real, pred05)
        RMSE20 =  calculate_rmse(real, pred20)

        print("DAS:", DA01, DA05, DA20)

        g=fig.to_json()
        g=json.loads(g)
        json.dumps(g)
        
        for i in g['data']:
        #    print(i['x'])
        #    print(i['y'])
            dic['Date']=i['x']
            dic['Price']=i['y']
        for i in range(len(dic['Date'])):
            dic['Date'][i]=dic['Date'][i][0:10]

        show=request.form['show']
        
        values, dates = get_results(ticker)
        pred_dic={}
        pred_dic['Date']=dates
        pred_dic['Price']=values
        #print(pred_dic['Date'])
        #print(pred_dic['Price'])

        return make_response(render_template('stock.html',
                date=dic['Date'],
                price=dic['Price'], 
                show=show, 
                ticker=ticker, 
                pred_date=pred_dic['Date'],
                pred_price=pred_dic['Price'],
                pred01=pred01,
                pred01_date=pred_dic['Date'],
                pred05=pred05,
                pred05_date=pred_dic['Date'],
                pred20=pred20,
                pred20_date=pred_dic['Date'],
                DA01=DA01, DA05=DA05, DA20=DA20,
                RMSE01=RMSE01, RMSE05=RMSE05, RMSE20=RMSE20
            )
        )

class Stock(Resource):
    def get(self):
        return make_response(render_template('stock.html'))
    def post(self):
        return make_response(render_template('stock.html'))

class Evaluation(Resource):
    def post(self):
        data = request.form
        ticker = data['ticker']
        diff = int(data['diff'])

        results = Results.query.filter_by(ticker=ticker).order_by(Results.pred_date.desc()).limit(41).all()

        pred = []
        real = []
        for result in results:
            pred.append(getattr(result, "d" + '{0:02d}'.format(diff)))
            real.append(result.d00)

        DA = calculate_DA(real, pred, diff) # Should be placed before shortenning list length to 20

        pred = pred[diff:20+diff]
        real = real[0:len(pred)]
        RMSE =  calculate_rmse(real, pred)

        return jsonify({'RMSE': RMSE, 'DA':DA})

def get_results(ticker):
    pred_date = datetime.today()
    pred_date = pred_date.date()
    result = Results.query.filter_by(ticker=ticker).order_by(Results.pred_date.desc()).first()
    
    values = []
    dates = []
    for i in range(1, 21):
        values.append(getattr(result, "d" + '{0:02d}'.format(i)))
        dates.append((datetime.today()+timedelta(i)).date().strftime('%Y-%m-%d'))

    return values, dates

def calculate_rmse(real, pred):
    real = np.array(real)
    pred = np.array(pred)
    rmse = np.sqrt(((pred - real) ** 2).mean())
    return round(rmse, 4)

#Directional Accuracy
def calculate_DA(real, pred, diff):
    pred_trends = []
    real_trends = []

    for i in range(len(pred)-diff):
        pred_trend = 1 if pred[i] - real[i] > 0 else 0
        pred_trends.append(pred_trend)
        real_trend = 1 if real[i] - real[i+diff] > 0 else 0
        real_trends.append(real_trend)

    cnt = 0
    for i in range(len(real_trends)):
        cnt += 1 if real_trends[i] == pred_trends[i] else 0

    DA = cnt/len(real_trends)

    print("PRED TRENDS: ", pred_trends)
    print("REAL TRENDS: ", real_trends)
    print("Directional Accuracy: ", DA * 100 , "%")

    return round(DA, 4)
