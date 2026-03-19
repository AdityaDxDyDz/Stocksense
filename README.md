📊 StockSense – AI Stock Market Prediction Platform

StockSense is a machine learning-powered web application that predicts stock price movements, analyzes market sentiment using news data, and provides an interactive dashboard for users.

🚀 Features

📈 Stock Price Prediction

Predicts next 7 days movement (UP/DOWN)

Uses ML models (Scikit-learn, XGBoost)

🧠 Market Regime Detection

Classifies market as:

Bull 🟢

Bear 🔴

Sideways 🟡

📰 News Sentiment Analysis

Fetches real-time financial news

Uses VADER sentiment analysis

📊 Interactive Dashboard

Recent predictions

Market sentiment score

Live TradingView charts

👤 User Authentication

Register/Login system

Role-based access (Admin/User)

⚙️ Admin Panel

Manage users

Monitor activity

Blacklist users

🏗️ Tech Stack

🔹 Backend

Flask

SQLAlchemy

Flask-Login

🔹 Machine Learning

Scikit-learn

XGBoost

Pandas, NumPy

🔹 Data & APIs

Yahoo Finance (yfinance)

NewsAPI

VADER Sentiment

🔹 Frontend

HTML, CSS, JavaScript

Chart.js

TradingView Widgets

📂 Project Structure

```
StockSense/
│
├── data/
│   ├── processed/
│   │   └── featured_data.csv        # Final dataset used for training
│   └── sp500_tickers.csv            # List of S&P 500 stock symbols
│
├── instance/
│   └── users.db                     # SQLite database for user authentication
│
├── models/
│   └── xgboost_model.pkl            # Trained XGBoost model file
│
├── nse_stock_data/                  # Raw/Processed NSE-specific CSVs
│
├── plots/                           # Generated visualizations (e.g., market regimes)
│
├── src/
│   ├── evaluate.py                  # Model performance evaluation metrics
│   ├── feature_engineering.py       # Technical indicators and data prep logic
│   ├── fetch_news.py                # News API and scraping integration
│   ├── main.py                      # Core backend logic / Entry point
│   ├── predict.py                   # Prediction inference script
│   ├── sentiment_analysis.py        # NLP for stock news sentiment
│   └── train_model.py               # XGBoost training pipeline
│
├── static/
│   ├── css/                         # Custom stylesheets (style.css)
│   └── js/                          # Frontend scripts
│
├── stock_data/                      # General historical stock data storage
│
├── templates/                       # Flask HTML templates (Jinja2)
│   ├── admin_panel.html
│   ├── dashboard.html
│   ├── home.html
│   ├── layout.html                  # Base template for UI consistency
│   ├── login.html
│   ├── predict.html
│   └── register.html
│
├── app.py                           # Main Flask application entry point
├── feature_nse.py                   # NSE-specific technical indicators
├── fetch_nse.py                     # NSE data retrieval script
├── models.py                        # SQLAlchemy database models
├── nse_stocks.csv                   # CSV containing NSE stock list
├── nse_tickers.py                   # Script for managing NSE ticker symbols
├── requirements.txt                 # Project library dependencies
└── README.md                        # Project documentation
```
⚙️ Installation & Setup
1️⃣ Clone the repository
```
git clone https://github.com/your-username/stocksense.git
cd stocksense
```
2️⃣ Create virtual environment
```
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```
3️⃣ Install dependencies
```
pip install -r requirements.txt
```
4️⃣ Add API Keys
Create a .env file:
```
NEWSAPI_KEY=your_newsapi_key
ALPHA_KEY=your_alpha_vantage_key
```
5️⃣ Run the application
```
python app.py
```
🧠 Machine Learning Pipeline

Data Collection (Yahoo Finance)

Feature Engineering

Model Training:

Linear Regression

Gradient Boosting

XGBoost

Model Evaluation

Prediction Generation

⚡ Performance Optimizations

Cached news & sentiment (reduces API latency)

Preloaded stock dataset (avoids repeated file reads)

Optimized database queries

Lightweight ML models

🔮 Future Improvements

🔄 Real-time stock updates

📡 Async background jobs (Celery/Redis)

📊 Advanced model ensemble

📱 Mobile responsiveness

🌐 Deployment (Render / AWS)

👨‍💻 Author

Aditya Raj

GitHub: https://github.com/AdityaDxDyDz

LinkedIn: https://linkedin.com/in/aditya-raj-3765582aa/
