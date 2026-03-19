from flask import Flask, render_template, request, redirect, url_for, abort, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from models import Prediction, db, User


app = Flask(__name__)

app.config["SECRET_KEY"] = "stocksense_secret"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# initialize database
db.init_app(app)


# setup login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


@app.route("/")
def home():

    # if current_user.is_authenticated:
    #     return redirect(url_for("dashboard"))

    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        userid = request.form["userid"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if not username or not userid or not password or not confirm_password:
            flash("All fields are required", "danger")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match", "danger")
            return redirect(url_for("register"))

        existing = User.query.filter_by(userid=userid).first()

        if existing:
            flash("⚠ UserID already exists", "error")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        user = User(
            username=username,
            userid=userid,
            password=hashed_password,
            role="user"
        )

        db.session.add(user)
        db.session.commit()

        flash("Registration successful! Please log in.", "success")

        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        userid = request.form["userid"]
        password = request.form["password"]

        user = User.query.filter_by(userid=userid).first()

        if user and check_password_hash(user.password, password):

            login_user(user)
            flash(f"Logged in successfully!, welcome {user.username}!", "success")

            if user.role == "admin":
                flash("Welcome Admin!", "info")
                return redirect(url_for("admin_panel"))
            else:
                return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials", "danger")
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


@app.route("/dashboard")
@login_required
def dashboard():

    users = User.query.all()

    import pandas as pd
    from src.predict import fetch_news
    from src.predict import get_news_sentiment

    news = fetch_news("Stock market OR global economy OR finance OR business")
    sentiment_score = get_news_sentiment(news)

    if sentiment_score > 0.05:
        market_regime = "Bullish 🟢"
    elif sentiment_score < -0.05:
        market_regime = "Bearish 🔴"
    else:
        market_regime = "Neutral 🟡"

    with open("stocks.txt") as f:
        stocks_list = [line.strip() for line in f]

    stocks = len(stocks_list)

    # top_prediction = get_top_prediction(stocks_list)

    recent_predictions = Prediction.query.filter_by(
        user_id=current_user.id
    ).order_by(Prediction.id.desc()).limit(5).all()
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()

    flash(f"Welcome to your dashboard, {current_user.username}!", "success")

    return render_template(
        "dashboard.html",
        users=users,
        stocks=stocks,
        total_predictions=total_predictions,
        predictions_list=recent_predictions,
        stocks_list=stocks_list,
        market_regime=market_regime,
        sentiment_score=round(sentiment_score, 2),
        news=news[:5]
    )

#for recent prediction from dashboard
@app.route("/predict/<symbol>")
@login_required
def predict_from_recent(symbol):

    from src.predict import predict_next_7_days

    result = predict_next_7_days(symbol)

    return render_template(
        "predict.html",
        symbol=symbol,
        predictions=result["predictions"],
        probabilities=result["probabilities"],
        labels=result["labels"],
        prices=result["prices"],
        history_dates=result["history_dates"],
        history_prices=result["history_prices"],
        regime=result["regime"],
        sentiment=result["sentiment"],
        news=result["news"],
        regime_chart=result["regime_chart"]
    )


@app.route("/admin")
@login_required
def admin_panel():
    from models import User, Prediction
    from sqlalchemy import func
    from datetime import datetime, timedelta

    if current_user.role != "admin":
        abort(403)

    users = User.query.filter(User.role != "admin").all()

    total_users = len(users)

    active_users = User.query.filter(
        User.role != "admin",
        User.is_blacklisted == False
    ).count()

    blacklisted_users = User.query.filter(
        User.role != "admin",
        User.is_blacklisted == True
    ).count()

    users_stats = [total_users, active_users, blacklisted_users]

    total_predictions = Prediction.query.count()

    days = []
    counts = []

    for i in range(5,0,-1):

        day = datetime.utcnow().date() - timedelta(days=i)

        count = Prediction.query.filter(
            func.date(Prediction.created_at) == day
        ).count()

        days.append(day.strftime("%a"))
        counts.append(count)


    return render_template(
        "admin_panel.html",
        users=users,
        total_users=total_users,
        active_users=active_users,
        blacklisted_users=blacklisted_users,
        total_predictions=total_predictions,
        users_stats=users_stats,
        trade_days=days,
        trade_counts=counts
    )


@app.route("/logout")
@login_required
def logout():

    logout_user()
    flash("Logged out successfully!", "success")
    return redirect(url_for("login"))

@app.route("/blacklist/<int:user_id>", methods=["POST"])
@login_required
def blacklist_user(user_id):

    if current_user.role != "admin":
        abort(403)

    user = User.query.get(user_id)

    user = User.query.get_or_404(user_id)
    user.is_blacklisted = True

    db.session.commit()
    flash(f"User {user.username} has been blacklisted.", "warning")
    return redirect(url_for("admin_panel"))

@app.route("/activate/<int:user_id>", methods=["POST"])
@login_required
def activate_user(user_id):

    if current_user.role != "admin":
        abort(403)

    user = User.query.get_or_404(user_id)

    user.is_blacklisted = False
    db.session.commit()
    flash(f"User {user.username} has been activated.", "success")
    return redirect(url_for("admin_panel"))

@app.route("/delete_user/<int:user_id>", methods=["POST"])
@login_required
def delete_user(user_id):

    if current_user.role != "admin":
        abort(403)

    user = User.query.get(user_id)

    db.session.delete(user)
    db.session.commit()
    flash(f"User {user.username} has been deleted.", "danger")
    return redirect(url_for("admin_panel"))

#predict
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():

    if request.method == "POST":

        symbol = request.form["symbol"].upper()

        from src.predict import predict_next_7_days
        result = predict_next_7_days(symbol)

        from models import Prediction
        prediction = Prediction(
            symbol=symbol,
            sentiment=result["sentiment"],
            regime=result["regime"],
            probability=max(result["probabilities"]),
            price=result["prices"][0],
            user_id=current_user.id
        )
        db.session.add(prediction)
        db.session.commit()

        top_prediction = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).limit(6).all()

        return render_template(
            "predict.html",
            symbol=symbol,
            predictions=result["predictions"],
            probabilities=result["probabilities"],
            labels=result["labels"],
            prices=result["prices"],
            history_dates=result["history_dates"],
            history_prices=result["history_prices"],
            regime=result["regime"],
            sentiment=result["sentiment"],
            news=result["news"],
            regime_chart=result["regime_chart"],
            top_prediction=top_prediction
        )
    return redirect(url_for("dashboard"))

if __name__ == "__main__":

    with app.app_context():
        db.create_all()

    app.run(debug=True)