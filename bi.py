import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Streamlit App
st.title("📊 LSTM Time Series Predictor")

# File Uploader
uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.write("### 📌 Preview of Dataset")
    st.write(df.head())

    # Ensure dataframe has numeric values
    st.write("### 📊 Dataset Summary")
    st.write(df.describe())

    # Select Target Column (Must be numeric)
    target_column = st.selectbox("🎯 Select Time Series Column for Prediction", df.columns)

    # Plot data
    st.write("### 📊 Time Series Visualization")
    plt.figure(figsize=(10, 5))
    plt.plot(df[target_column], label=target_column)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    st.pyplot(plt)

    # Normalize Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[[target_column]])

    # Create sequences for LSTM
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    seq_length = st.slider("📏 Select Sequence Length", min_value=5, max_value=100, value=20)
    X, y = create_sequences(df_scaled, seq_length)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape input for LSTM (samples, timesteps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # LSTM Model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train Model
    if st.button("🚀 Train LSTM Model"):
        model.fit(X_train, y_train, batch_size=16, epochs=20, verbose=1)

        # Predict on Test Data
        y_pred = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred)

        # Show Results
        st.write("### 📊 Predicted vs Actual")
        plt.figure(figsize=(10, 5))
        plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label="Actual")
        plt.plot(y_pred, label="Predicted")
        plt.legend()
        st.pyplot(plt)

        st.write("✅ Model Training Completed!")

    # Predict Future Data
    st.write("### 🔮 Make Future Predictions")
    future_steps = st.slider("📏 Predict Future Steps", min_value=5, max_value=100, value=10)

    if st.button("🔮 Predict Future Values"):
        last_seq = df_scaled[-seq_length:]
        future_preds = []

        for _ in range(future_steps):
            last_seq = np.reshape(last_seq, (1, seq_length, 1))
            pred = model.predict(last_seq)
            future_preds.append(pred[0][0])
            last_seq = np.append(last_seq[:, 1:, :], [[pred]], axis=1)

        future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

        # Plot Future Predictions
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(df[target_column]), len(df[target_column]) + future_steps), future_preds, marker='o', linestyle="dashed", label="Future Predictions")
        plt.legend()
        st.pyplot(plt)

        st.write("✅ Future Predictions Complete!")
