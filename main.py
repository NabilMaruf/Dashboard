# Import library
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Prediksi EV Stock", layout="centered")

# Header
st.markdown("""
    <div style='background-color: #004d99; padding: 20px; border-radius: 10px;'>
        <h1 style='text-align: center; color: white;'>🌍 EV Stock Global Prediction</h1>
        <p style='text-align: center; color: #cce6ff; font-size: 18px;'>
            Muhammad Nabil Ma'ruf | Sistem Informasi A | 2309116035
        </p>
    </div>
    <br>
""", unsafe_allow_html=True)


# Sidebar
st.sidebar.header("⚙️ Pengaturan")
model_choice = st.sidebar.selectbox("📌 Pilih Model", ("Random Forest", "XGBoost"))

# Load Data
data_url = "https://raw.githubusercontent.com/MuhammadRofif/abc/refs/heads/main/Final_Data.csv"
try:
    data = pd.read_csv(data_url)
    st.sidebar.success("✅ Data berhasil dimuat.")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat data: {e}")
    st.stop()

# Data Preparation
ev_stock = data[data['parameter'] == 'EV stock']
ev_stock = ev_stock[ev_stock['value'] > 0]
ev_stock = ev_stock[(np.abs(stats.zscore(ev_stock['value'])) < 3)]

X = ev_stock[['region', 'mode', 'powertrain', 'year']]
y = ev_stock['value']

# Encoding
le_region = LabelEncoder()
le_mode = LabelEncoder()
le_powertrain = LabelEncoder()
X['region'] = le_region.fit_transform(X['region'])
X['mode'] = le_mode.fit_transform(X['mode'])
X['powertrain'] = le_powertrain.fit_transform(X['powertrain'])

# Scaling
scaler = StandardScaler()
X['year'] = scaler.fit_transform(X[['year']])

# Log Transform target
y = np.log1p(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeling
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Prediction
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

# Scatter Plot Actual vs Predicted
st.subheader(f"📈 Scatter Plot: Actual vs Predicted ({model_choice})")
fig1, ax1 = plt.subplots(figsize=(8,6))
if model_choice == "Random Forest":
    ax1.scatter(np.expm1(y_test), np.expm1(rf_pred), alpha=0.5, color='dodgerblue')
    ax1.set_ylabel('Predicted EV Stock (RF)')
else:
    ax1.scatter(np.expm1(y_test), np.expm1(xgb_pred), alpha=0.5, color='darkorange')
    ax1.set_ylabel('Predicted EV Stock (XGB)')

ax1.set_xlabel('Actual EV Stock')
ax1.set_title(f'{model_choice}: Actual vs Predicted EV Stock')
ax1.grid(True)
st.pyplot(fig1)

# Feature Importance
st.subheader(f"🏆 Feature Importance: {model_choice}")
if model_choice == "Random Forest":
    importances = rf_model.feature_importances_
else:
    importances = xgb_model.feature_importances_

feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

fig2, ax2 = plt.subplots(figsize=(8,6))
feat_imp.plot(kind='bar', ax=ax2, color='mediumseagreen')
ax2.set_title(f'{model_choice} Feature Importance')
ax2.set_ylabel('Importance Score')
ax2.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig2)

# Footer
st.markdown("<hr style='border: 1px solid #2E8B57;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 12px;'>© 2025 - Muhammad Nabil Ma'ruf | Sistem Informasi A 📘</p>", unsafe_allow_html=True)

