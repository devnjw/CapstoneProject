from flask import render_template, make_response, request, flash, session, g, jsonify
from flask_restful import Resource, abort
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta

import pandas as pd
import pandas_datareader as web
import json
import plotly.express as px
import numpy as np
from app.models import User, Results, TickerInfo
from .. import db

class Home(Resource):
    def get(self):
        return make_response(render_template('index.html'))

    def post(self):
        
        input_stock = request.form['ticker']
        Ticker = TickerInfo.query.filter_by(ticker=input_stock).first()
        if not Ticker:
            Ticker = TickerInfo.query.filter_by(kor_name=input_stock).first()
        if not Ticker:
            Ticker = TickerInfo.query.filter_by(eng_name=input_stock).first()
        if not Ticker:
            flash('Database에 존재하지 않는 종목입니다.')
            return make_response(render_template('index.html'))
        ticker = Ticker.ticker

        print("Ticker: ", ticker)


        yesterday = datetime.today() - timedelta(1)
        past365 = datetime.today() - timedelta(30)

        yesterday = yesterday.strftime('%Y-%m-%d')
        past365 = past365.strftime('%Y-%m-%d')
        values, dates = get_results(ticker)
        df = web.DataReader(f'{ticker}.KS', data_source='yahoo', start=f'{past365}', end=f'{yesterday}')
        df=df.reset_index()
        
        fig = px.line(df, x='Date', y='Close', title='Time Series with Rangeslider')
        dic = {}
        fig.update_xaxes(rangeslider_visible=True)
        g=fig.to_json()
        g=json.loads(g)
        json.dumps(g)
        

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

        if(len(pred20)>=20):
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
        else:
            DA01 = None
            DA05 = None
            DA20 = None
            RMSE01 = None
            RMSE05 = None
            RMSE20 = None
            

        #print("DAS:", DA01, DA05, DA20)

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
        print(pred_dic['Date'])
        print(pred_dic['Price'])

        ticker = Ticker.kor_name + '(' + Ticker.ticker + ')'

        return make_response(render_template('index.html',
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

def get_results(ticker):
    pred_date = datetime.today()
    pred_date = pred_date.date()
    result = Results.query.filter_by(ticker=ticker).order_by(Results.pred_date.desc()).first()
    values = []
    dates = []
    
    j=0
    for i in range(20):
        if(len(dates)>=20):
            break
        print((datetime.today()+timedelta(j)).date().weekday())
        if((datetime.today()+timedelta(j)).date().weekday()==5):
            j=j+2
        values.append(getattr(result, "d" + '{0:02d}'.format(i)))
        dates.append((datetime.today()+timedelta(j)).date().strftime('%Y-%m-%d'))
        print(dates)
        j=j+1

    # for i in range(0, 20):
    #     values.append(getattr(result, "d" + '{0:02d}'.format(i)))
    #     dates.append((datetime.today()+timedelta(i)).date().strftime('%Y-%m-%d'))
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

    return round(DA, 4)

class Ranking(Resource):
    def get(self, interval):
        pred_date = datetime.datetime.utcnow()
        if interval is 1:
            rank_list = Results.query.filter_by(pred_date=pred_date).order_by(Results.d01.desc())
        elif interval is 5:
            rank_list = Results.query.filter_by(pred_date=pred_date).order_by(Results.d05.desc())
        elif interval is 20:
            rank_list = Results.query.filter_by(pred_date=pred_date).order_by(Results.d20.desc())
        return make_response(render_template('home.html'))

class About(Resource):
    def get(self):
        return make_response(render_template('about.html'))

class Services(Resource):
    def get(self):
        return make_response(render_template('services.html'))

class Contact(Resource):
    def get(self):
        return make_response(render_template('contact.html'))

class Popup(Resource):
    def get(self):
        return make_response(render_template('popup.html'))
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


        yesterday = datetime.today() - timedelta(1)
        past365 = datetime.today() - timedelta(30)

        yesterday = yesterday.strftime('%Y-%m-%d')
        past365 = past365.strftime('%Y-%m-%d')
        values, dates = get_results(ticker)
        df = web.DataReader(f'{ticker}.KS', data_source='yahoo', start=f'{past365}', end=f'{yesterday}')
        df=df.reset_index()
        
        fig = px.line(df, x='Date', y='Close', title='Time Series with Rangeslider')
        dic = {}
        fig.update_xaxes(rangeslider_visible=True)
        g=fig.to_json()
        g=json.loads(g)
        json.dumps(g)
        

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
        print(pred_dic['Price'])

        return make_response(render_template('popup.html',
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

class ArtificialRanking(Resource):
    def get(self):
        try:
            for i in range(60):
                ticker = "005930"
                day_diff = i
                print("Ticker: ", ticker)

                pred_date = datetime.today() - timedelta(int(day_diff))
                past_date = datetime.today() - timedelta(int(day_diff) + 40)

                pred_date = pred_date.strftime('%Y-%m-%d')
                past_date = past_date.strftime('%Y-%m-%d')
                
                df = web.DataReader(f'{ticker}.KS', data_source='yahoo', start=f'{past_date}', end=f'{pred_date}')
                df=df.reset_index()

                new_result = Results()
                new_result.pred_date = pred_date
                new_result.ticker = ticker

                for i in range(21):
                    setattr(new_result, "d" + '{0:02d}'.format(i), df['Close'][i] )

                db.session.add(new_result)
                db.session.commit()

        except Exception as e:
            print(e)
            abort(500)

        return jsonify({'result': "Success"})

    def post(self):
        try:
            ticker = request.form['ticker']
            day_diff = request.form['diff']
            print("Ticker: ", ticker)

            pred_date = datetime.today() - timedelta(int(day_diff))
            past_date = datetime.today() - timedelta(int(day_diff) + 40)

            pred_date = pred_date.strftime('%Y-%m-%d')
            past_date = past_date.strftime('%Y-%m-%d')
            
            df = web.DataReader(f'{ticker}.KS', data_source='yahoo', start=f'{past_date}', end=f'{pred_date}')
            df=df.reset_index()

            print("TESTSETS")

            new_result = Results()
            new_result.pred_date = pred_date
            new_result.ticker = ticker

            for i in range(21):
                setattr(new_result, "d" + '{0:02d}'.format(i), df['Close'][i] )

            db.session.add(new_result)
            db.session.commit()

        except Exception as e:
            print(e)
            abort(500)

        return jsonify({'result': "Success"})