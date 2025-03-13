import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Streamlit App Title
st.title("📊 Data Analyzer & Predictor")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset:")
    st.write(df.head())

    # Show basic statistics
    st.write("### Basic Statistics:")
    st.write(df.describe())

    # Show missing values
    st.write("### Missing Values:")
    st.write(df.isnull().sum())
    
    # Data Visualization
    st.write("## 📈 Data Visualization")
    selected_col = st.selectbox("Select a column for visualization", df.columns)
    if df[selected_col].dtype in ['int64', 'float64']:
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.write("Selected column is not numerical.")
    
    # Model Training
    st.write("## 🤖 Machine Learning Prediction")
    target = st.selectbox("Select target column", df.columns)
    features = [col for col in df.columns if col != target]
    
    if st.button("Train Model"):
        X = df[features].select_dtypes(include=['number'])  # Selecting only numerical features
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        st.write(f"### Model Accuracy: {acc:.2f}")