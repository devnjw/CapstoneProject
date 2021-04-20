from flask import Flask
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

    # Routing
    from .views.machine_views import PredictMachine, TrainMachine, Scheduler
    api.add_resource(Scheduler, '/scheduler')
    api.add_resource(PredictMachine, '/predict')
    api.add_resource(TrainMachine, '/research')


    return app