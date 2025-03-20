import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist

# ✅ Load trained XGBoost model correctly
@st.cache_data
def load_model():
    model = xgb.Booster()
    model.load_model(r"ImagoAI_Assignment\xgboost_model.json")  # Ensure this file exists in the directory
    return model

model = load_model()  # Load the model

# ✅ SNV Normalization
def snv(input_data):
    return (input_data - np.mean(input_data, axis=1, keepdims=True)) / np.std(input_data, axis=1, keepdims=True)

# ✅ Successive Projections Algorithm (SPA)
def successive_projections_algorithm(X, num_features):
    num_samples, num_wavelengths = X.shape
    selected_features = []
    remaining_features = list(range(num_wavelengths))
    
    first_feature = np.argmax(np.var(X, axis=0))
    selected_features.append(first_feature)
    remaining_features.remove(first_feature)

    for _ in range(num_features - 1):
        distances = cdist(X[:, selected_features].T, X[:, remaining_features].T, metric='euclidean')
        next_feature = remaining_features[np.argmax(np.min(distances, axis=0))]
        selected_features.append(next_feature)
        remaining_features.remove(next_feature)
    
    return selected_features

# ✅ Streamlit UI
st.title("🌽 Mycotoxin Prediction using Hyperspectral Imaging")
st.write("Upload a CSV file containing spectral reflectance data for mycotoxin level prediction.")

# 📂 File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 **Uploaded Data Preview:**")
    st.write(df.head())

    # Extract features (removing the first column which is the sample ID)
    X = df.iloc[:, 1:].values  

    # ✅ Apply Preprocessing
    st.write("🔄 **Applying Preprocessing (SNV + Savitzky-Golay Filtering + SPA)...**")
    X_snv = snv(X)
    X_sg = savgol_filter(X_snv, window_length=5, polyorder=2, axis=1)

    # ✅ Apply SPA (use 20 selected features)
    num_selected_features = 20
    selected_features = successive_projections_algorithm(X_sg, num_selected_features)
    X_final = X_sg[:, selected_features]

    # Convert to DataFrame for final predictions
    df_final = pd.DataFrame(X_final, columns=[f"Wavelength_{i}" for i in selected_features])

    # ✅ Convert input features to XGBoost DMatrix format before predicting
    dmatrix = xgb.DMatrix(df_final)

    # ✅ Predict using loaded XGBoost model
    y_pred = model.predict(dmatrix)

    # ✅ Display Results
    df["Predicted Mycotoxin Levels"] = y_pred
    st.write("📈 **Predictions:**")
    st.write(df)

    # 📥 Download Button for Predictions
    df.to_csv("ImagoAI_Assignment/predictions.csv", index=False)
    st.download_button("📥 Download Predictions", "predictions.csv", "text/csv")
