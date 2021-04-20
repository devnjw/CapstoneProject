from flask_restful import Resource
from datetime import datetime, timedelta

import pandas as pd
import pandas_datareader as web
import numpy as np

from app.models import Results, BestModels
from .. import db

def save_result(results, ticker):
    pred_date=datetime.utcnow()

    df = web.DataReader(f'{ticker}.KS', data_source='yahoo', start=f'{pred_date-timedelta(5)}', end=f'{pred_date}')
    df.sort_index(ascending=False)

    results = results.astype(int).tolist()

    print("RESULTS:", results)

    new_result = Results()
    new_result.pred_date = pred_date
    new_result.ticker = ticker
    new_result.d00 = df['Close'][0]
    for i in range(1, 21):
        setattr(new_result, "d" + '{0:02d}'.format(i), results[i-1] )

    db.session.add(new_result)
    db.session.commit()

def save_parameters(parameters):
    print("Params:", parameters)
    db.session.add(parameters)
    db.session.commit()
    return parameters.id

def save_best_model(ticker, filepath, params_id):
    model = BestModels.query.filter_by(ticker=ticker).first()
    if model:
        model.filepath = filepath
        model.params_id = params_id
        db.session.commit()
        print("Update New Model Success")
    else:
        new_model = BestModels(
            ticker=ticker,
            filepath=filepath,
            params_id=params_id
        )
        db.session.add(new_model)
        db.session.commit()
        print("Save New Model Success")