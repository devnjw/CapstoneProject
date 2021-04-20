from flask import request, jsonify, render_template, make_response
from flask_restful import Resource
from datetime import datetime, timedelta

from app.models import Results
from app.machine.scheduler import run_scheduler, Func
from app.machine.predict import predict
from app.machine.train import train
from .. import db

class Scheduler(Resource):
    def get(self):
        # Func()
        run_scheduler("00:11")
        return jsonify({'result': "Scheduler Ends Successfully"})

    def post(self):
        Func()
        return jsonify({'result': "Scheduler Ends Successfully"})


class PredictMachine(Resource):
    def get(self):
        return make_response(render_template('admin_predict.html'))

    def post(self):
        data = request.form
        predict()
        return jsonify({'result': "Predict Success"})

class TrainMachine(Resource):
    def get(self):
        return make_response(render_template('admin_train.html'))

    def post(self):
        data = request.form
        train()
        return jsonify({'result': "Train Success"})