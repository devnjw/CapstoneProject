from app import db

class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(100))
    email = db.Column(db.String(200))
    registered_on = db.Column(db.DateTime, nullable=False)
    cnt_login = db.Column(db.Integer)

class BestModels(db.Model):
    __tablename__ = "best_models"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ticker = db.Column(db.String(20))
    filepath = db.Column(db.String(255))
    params_id = db.Column(db.Integer, db.ForeignKey('parameters.id'))

class MyList(db.Model):
    __tablename__ = "my_list"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50))
    ticker = db.Column(db.String(20))
    saved_on = db.Column(db.DateTime, nullable=False)

class Parameters(db.Model):
    __tablename__ = "parameters"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ticker = db.Column(db.String(20))
    start_date = db.Column(db.DateTime) # Train dataset Start Date
    end_date = db.Column(db.DateTime) # Train dataset End Date
    pred_date = db.Column(db.DateTime) # Predicted Date

    RMSE = db.Column(db.Float)
    MAPE = db.Column(db.Float)
    DA = db.Column(db.Float) # Directional Accuracy
    OR = db.Column(db.Float) # Optimism Ratios
    PR = db.Column(db.Float) # Pessimism Ratios

    hidden_layer_size = db.Column(db.Integer)
    stack_size = db.Column(db.Integer)
    train_window = db.Column(db.Integer)
    learning_rate = db.Column(db.Float)
    epoch = db.Column(db.Integer)

class TickerInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ticker = db.Column(db.String(20))
    eng_name = db.Column(db.String(50))
    kor_name = db.Column(db.String(50))
    
class Ranking(db.Model):
    __tablename__ = "ranking"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ticker = db.Column(db.String(20))
    kor_name = db.Column(db.String(50))
    curr_price = db.Column(db.Integer)
    pred_price = db.Column(db.Integer)
    fluctuation = db.Column(db.Float)

class Results(db.Model):
    __tablename__ = "results"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    pred_date = db.Column(db.Date, nullable=False)
    ticker = db.Column(db.String(20))
    d00 = db.Column(db.Integer)
    d01 = db.Column(db.Integer)
    d02 = db.Column(db.Integer)
    d03 = db.Column(db.Integer)
    d04 = db.Column(db.Integer)
    d05 = db.Column(db.Integer)
    d06 = db.Column(db.Integer)
    d07 = db.Column(db.Integer)
    d08 = db.Column(db.Integer)
    d09 = db.Column(db.Integer)
    d10 = db.Column(db.Integer)
    d11 = db.Column(db.Integer)
    d12 = db.Column(db.Integer)
    d13 = db.Column(db.Integer)
    d14 = db.Column(db.Integer)
    d15 = db.Column(db.Integer)
    d16 = db.Column(db.Integer)
    d17 = db.Column(db.Integer)
    d18 = db.Column(db.Integer)
    d19 = db.Column(db.Integer)
    d20 = db.Column(db.Integer)

