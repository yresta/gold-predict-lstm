import streamlit as st 
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Import ManualLSTM kalau diperlukan untuk custom model
from manual_lstm import ManualLSTM  

# Load model dan scaler
scaler_X = joblib.load("scaler_x.pkl")
scaler_y = joblib.load("scaler_y.pkl")
model_manual_lstm_loaded = load_model(
    "manual_model_lstm.keras",
    custom_objects={"ManualLSTM": ManualLSTM}
)

# Fungsi membersihkan angka yang pakai titik ribuan dan koma desimal
def clean_numeric_columns(df, cols):
    for col in cols:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace('.', '', regex=False)       # hapus titik (ribuan)
        df[col] = df[col].str.replace(',', '.', regex=False)      # ubah koma jadi titik (desimal)
        df[col] = df[col].astype(float)                           # ubah ke float
    return df

# Load data dan bersihkan angka
@st.cache_data
def load_data():
    df = pd.read_csv("DataEmas.csv")
    df.rename(columns={
        "Tanggal": "Date",
        "Pembukaan": "Open",
        "Tertinggi": "High",
        "Terendah": "Low",
        "Terakhir": "Close",
        "Vol.": "Volume",
        "Perubahan%": "Change%"
    }, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])  # baru parse kolom yang sudah di-rename
    df = clean_numeric_columns(df, ['Open', 'High', 'Low'])  # bersihkan kolom angka
    return df

# Fungsi prediksi harga
def predict_future_price(df, model, scaler_X, scaler_y, future_date_str, time_steps=7):
    try:
        df = df.sort_values("Date").reset_index(drop=True)
        last_date = df["Date"].iloc[-1]
        future_date = pd.to_datetime(future_date_str)

        if future_date <= last_date:
            return f"⚠️ Tanggal {future_date_str} sudah ada atau kurang dari data terakhir."

        delta_days = (future_date - last_date).days
        if delta_days <= 0:
            return f"⚠️ Tanggal tidak valid."

        # Ambil window data terakhir
        past_window = df[['Open', 'High', 'Low']].iloc[-time_steps:].values
        scaled_window = scaler_X.transform(pd.DataFrame(past_window, columns=['Open', 'High', 'Low']))

        for _ in range(delta_days):
            input_seq = np.expand_dims(scaled_window, axis=0)
            pred_scaled = model.predict(input_seq, verbose=0)
            pred = scaler_y.inverse_transform(pred_scaled)[0][0]

            # Tambahkan prediksi ke window berikutnya
            next_input = np.array([[pred, pred, pred]])
            next_input_scaled = scaler_X.transform(pd.DataFrame(next_input, columns=['Open', 'High', 'Low']))
            scaled_window = np.vstack([scaled_window[1:], next_input_scaled])

        return f"💰 Prediksi harga emas pada **{future_date_str}**: **{pred:.2f}**"

    except Exception as e:
        return f"❌ Terjadi error saat prediksi: {e}"

# ========================
# 🖥️ STREAMLIT APP START
# ========================
st.title("📈 Prediksi Harga Emas Menggunakan Manual LSTM")

df = load_data()

st.subheader("🗓️ Pilih Tanggal Prediksi")
user_input_date = st.date_input("Pilih tanggal (setelah data terakhir)", 
                                min_value=df["Date"].max().date() + pd.Timedelta(days=1))

if st.button("🔮 Prediksi"):
    hasil = predict_future_price(df, model_manual_lstm_loaded, scaler_X, scaler_y, str(user_input_date))
    st.success(hasil)

st.markdown("---")
st.caption("🧠 Model LSTM manual dengan data time series harga emas.")
