from flask import request, jsonify, render_template, make_response
from flask_restful import Resource

class KPI(Resource):
    def get(self):
        return make_response(render_template('index.html'))

class TrainStats(Resource):
    def get(self):
        return make_response(render_template('TrainStats.html'))