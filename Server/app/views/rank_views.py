from flask import render_template, make_response, request, flash, session, g, jsonify
from flask_restful import Resource, abort
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta

import pandas as pd
import pandas_datareader as web

from app.models import Results, Ranking, TickerInfo
from .. import db

class Rank(Resource):
    def post(self):
        print("Method: GET")
        stock_rank = Ranking.query.order_by(Ranking.fluctuation.desc())
        stock_rank_down = Ranking.query.order_by(Ranking.fluctuation)
        return make_response(render_template('rank.html', stock_rank=stock_rank, stock_rank_down=stock_rank_down))

    def get(self):
        interval = 1

        print("input interval: ", interval)

        today = datetime.today()
        yesterday = datetime.today() - timedelta(5)

        today = today.strftime('%Y-%m-%d')
        yesterday = yesterday.strftime('%Y-%m-%d')

        try:
            df = web.DataReader(f'005930.KS', data_source='yahoo', start=yesterday, end='2021.04.08')
            df=df.reset_index()
        except Exception as e:
            print(e)
            return make_response(render_template('rank.html'))
        

        print(df['Date'][0].date())

        if interval is 1:
            stock_ranks = Results.query\
                .distinct(Results.ticker)\
                .join(TickerInfo, Results.ticker==TickerInfo.ticker)\
                .add_columns(TickerInfo.ticker, TickerInfo.kor_name, Results.d00, Results.d01, Results.pred_date)\
                .order_by(Results.pred_date.desc())\
                .order_by(Results.d01 - Results.d00).limit(10)\
                .all()

            stock_ranks_down = Results.query\
                .distinct(Results.ticker)\
                .join(TickerInfo, Results.ticker==TickerInfo.ticker)\
                .add_columns(TickerInfo.ticker, TickerInfo.kor_name, Results.d00, Results.d01, Results.pred_date)\
                .order_by(Results.pred_date.desc())\
                .order_by((Results.d01 - Results.d00).desc()).limit(10)\
                .all()

        elif interval is 5:
            stock_ranks = Results.query.filter_by(pred_date=df['Date'][0].date()).order_by(Results.d05.desc())
            stock_ranks_down = Results.query.filter_by(pred_date=df['Date'][0].date()).order_by(Results.d05)
        elif interval is 20:
            stock_ranks = Results.query.filter_by(pred_date=df['Date'][0].date()).order_by(Results.d20.desc())
            stock_ranks_down = Results.query.filter_by(pred_date=df['Date'][0].date()).order_by(Results.d20)


        results = []
        results_down = []


        for stock_rank in stock_ranks:
            print("STOCK RANK", stock_rank)
            rank = Ranking(
                ticker=stock_rank[1],
                kor_name=stock_rank[2],
                curr_price=stock_rank[3],
                pred_price=stock_rank[4],
                fluctuation=stock_rank[3]-stock_rank[4]
            )
            results.append(rank)
        
        for stock_rank in stock_ranks_down:
            print("STOCK RANK", stock_rank)
            rank = Ranking(
                ticker=stock_rank[1],
                kor_name=stock_rank[2],
                curr_price=stock_rank[3],
                pred_price=stock_rank[4],
                fluctuation=stock_rank[3]-stock_rank[4]
            )
            results_down.append(rank)

        print("STOCK", stock_rank)
        return make_response(render_template('rank.html', stock_rank=results, stock_rank_down=results_down))