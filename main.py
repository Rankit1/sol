from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import kagglehub
import google.generativeai as genai

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow frontend to access the backend from any domain

# Load and preprocess dataset
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.sort_values(by=["Date"])
    df.dropna(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df[df.index >= "2021-01-01"]

    encoder = LabelEncoder()
    df["Stock"] = encoder.fit_transform(df["Stock"])

    scaler = MinMaxScaler()
    df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

    return df, encoder, scaler

# Prepare dataset for LSTM
def prepare_dataset(df, time_steps=20):
    def create_sequences(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i : i + time_steps])
            y.append(data[i + time_steps, 3])  # Predict "Close" price
        return np.array(X), np.array(y)

    X_list, y_list = [], []
    for stock_id in df["Stock"].unique():
        stock_data = df[df["Stock"] == stock_id].drop(["Stock"], axis=1).values
        if len(stock_data) > time_steps:
            X_stock, y_stock = create_sequences(stock_data, time_steps)
            X_list.append(X_stock)
            y_list.append(y_stock)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y

# Build and train LSTM model
def build_and_train_model(X, y, time_steps=20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    model = Sequential([
        Input(shape=(time_steps, X.shape[2])),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model.fit(X_train[-100000:], y_train[-100000:], epochs=10, batch_size=64, validation_data=(X_test, y_test))

    return model

# AI-generated insights using Gemini
def generate_gemini_insights(y_pred_sample, stock_name):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""
    Analyze future trends for stock: {stock_name}
    Predicted Prices: {y_pred_sample}
    Provide investment insights and actionable recommendations.
    """

    response = model.generate_content(prompt)
    return response.text

# Flask Routes
@app.route('/predict_stock', methods=['GET'])
def predict_stock():
    stock_name = request.args.get('stock_name')
    if not stock_name:
        return jsonify({"error": "Stock name is required"}), 400

    result = generate_stock_insights(stock_name, df, encoder, scaler, model)
    return jsonify(result)

@app.route('/market_insights', methods=['GET'])
def market_insights():
    result = generate_market_insights(df, encoder, scaler, model)
    return jsonify(result)

# Load dataset and train model on startup
if __name__ == '__main__':
    print("Downloading dataset...")
    path = kagglehub.dataset_download("andrewmvd/india-stock-market")

    # Select specific CSV file
    filepath = None
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename == "stocks_df.csv":
                filepath = os.path.join(dirname, filename)
                break
        if filepath:
            break

    if not filepath:
        raise FileNotFoundError("stocks_df.csv not found in downloaded dataset.")

    print(f"Using dataset file: {filepath}")

    # Load, preprocess, train
    df, encoder, scaler = load_and_preprocess_data(filepath)
    X, y = prepare_dataset(df)
    model = build_and_train_model(X, y)

    app.run(host='0.0.0.0', port=8080,debug=True)

