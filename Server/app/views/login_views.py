from flask import render_template, make_response, request, flash, session, g
from flask_restful import Resource
from werkzeug.security import generate_password_hash, check_password_hash
import datetime

from app.models import User
from .. import db

class Login_function(Resource):
    def get(self):
        return make_response(render_template('login.html'))

    def post(self):
        data = request.form
        user = User.query.filter_by(username=data['username']).first()

        if not user:
            flash('존재하지 않는 사용자입니다.')
        elif not check_password_hash(user.password, data['password']):
            flash('비밀번호가 올바르지 않습니다.')
        else:
            #user.cnt_login += 1
            db.session.commit()
            return make_response(render_template('stock.html'))
            
        return make_response(render_template('login.html'))

class SignUp_function(Resource):
    def post(self):
        data = request.form
        user = User.query.filter_by(username=data['username']).first()
        if not user:
            new_user = User(
                #email=data['email'],
                username=data['username'],
                email=data['email'],
                password=generate_password_hash(data['password']),
                registered_on=datetime.datetime.utcnow(),
                cnt_login=0
            )
            save_changes(new_user)
            flash('환영합니다.\n가입이 완료되었습니다.')
            return make_response(render_template('login.html'))
        else:
            flash('이미 존재하는 사용자입니다.')
            return make_response(render_template('login.html'))

def userlist():
    user_list = User.query.order_by(User.registered_on.desc())
    return user_list


# 아래 :, -> 는 python function annotation이다.
# python에서 type을 지정해주지 않아서 발생할 수 있는 버그를 잡기 쉽게 도와준다.
# 런타임에서는 아무 일도 일어나지 않고, IDE, mypy를 사용해 검사할 때만 작동하는 주석이다.
def save_changes(data: User) -> None:
    db.session.add(data)
    db.session.commit()