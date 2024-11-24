from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import yfinance as yf

# Initialize Flask app
app = Flask(__name__)

# Step 1: Data Collection
def get_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df

# Step 2: Feature Engineering
def create_features(df):
    df['Price_Change'] = df['Close'].shift(-1) - df['Close']
    df['Target'] = np.where(df['Price_Change'] > 0, 1, 0)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['Volatility'] = df['Close'].rolling(window=5).std()
    df.dropna(inplace=True)
    return df[['Close', 'SMA_5', 'SMA_10', 'Volatility', 'Target']]

# Step 3: Model Training
def train_model(df):
    X = df[['Close', 'SMA_5', 'SMA_10', 'Volatility']]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    symbol = request.form['symbol']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    # Fetch and process data
    data = get_data(symbol, start_date, end_date)
    features = create_features(data)

    # Train model
    model, X_test, y_test = train_model(features)

    # Predict the latest movement
    latest_data = features.tail(1).drop(columns=['Target'])  # Ensure only features are used
    prediction = model.predict(latest_data)

    # Prepare prediction result
    result = 'Up' if prediction[0] == 1 else 'Down'

    return render_template('result.html', symbol=symbol, result=result)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)