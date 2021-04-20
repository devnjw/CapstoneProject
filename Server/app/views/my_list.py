from flask import render_template, make_response, request, flash, session, g
from flask_restful import Resource
from werkzeug.security import generate_password_hash, check_password_hash
import datetime

from app.models import User
from .. import db

class MyList(Resource):
    def get(self, username):
        return make_response(render_template('login.html'))