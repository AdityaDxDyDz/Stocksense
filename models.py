from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()


class User(UserMixin, db.Model):

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(
        db.String(100),
        nullable=False
    )

    userid = db.Column(
        db.String(100),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(200),
        nullable=False
    )

    role = db.Column(
        db.String(20),
        default="user"
    )

    is_blacklisted = db.Column(
        db.Boolean,
        default=False
    )

    def __repr__(self):
        return f"<User {self.userid}>"
    
class Prediction(db.Model):

    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)

    symbol = db.Column(
        db.String(20),
        nullable=False
    )

    sentiment = db.Column(
        db.Float
    )

    regime = db.Column(
        db.String(50)
    )

    created_at = db.Column(
        db.DateTime,
        default=db.func.current_timestamp()
    )

    probability = db.Column(
        db.Float
    )

    price = db.Column(
        db.Float
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id")
    )

    def __repr__(self):
        return f"<Prediction {self.symbol}>"