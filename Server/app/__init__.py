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

    # Routing
    from .views.login_views import Login_function, SignUp_function
    # api.add_resource(Login_function, '/')
    api.add_resource(SignUp_function, '/signup')

    from .views.home_views import Home, ArtificialRanking, About, Services, Contact, Popup
    api.add_resource(Home, '/')
    api.add_resource(About, '/about')
    api.add_resource(Services, '/services')
    api.add_resource(Contact, '/contact')
    api.add_resource(Popup, '/popup')
    api.add_resource(ArtificialRanking, '/home/ArtificialRanking')

    from .views.rank_views import Rank
    api.add_resource(Rank, '/rank')
    
    from .views.pred_view import Predict, Stock, Evaluation
    api.add_resource(Predict, '/pred')
    api.add_resource(Stock,'/stock')
    api.add_resource(Evaluation, '/evaluation')

    return app