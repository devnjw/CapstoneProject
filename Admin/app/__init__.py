from flask import Flask, request, make_response
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_restful import Resource, Api

from .config import BaseConfig

db = SQLAlchemy()
migrate = Migrate(compare_type=True)

def create_app():
    app = Flask(__name__)
    app.config.from_object(BaseConfig)
    api = Api(app)

    # ORM
    db.init_app(app)
    migrate.init_app(app, db)

    from . import models

    from .research_views import KPI, TrainStats
    api.add_resource(KPI, '/')
    api.add_resource(TrainStats, '/TrainStats')

    return app