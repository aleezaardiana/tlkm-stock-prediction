import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

# Konfigurasi halaman Streamlit
st.set_page_config(layout="wide")
st.title("Prediksi Harga Saham TLKM")

# Ambil data TLKM 5 tahun terakhir dari Yahoo Finance
data = yf.download('TLKM.JK', period='5y', auto_adjust=False, group_by='ticker')

# Periksa dan perbaiki kolom bertingkat
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(0)

# Jika data kosong, hentikan aplikasi
if data.empty:
    st.error("Data tidak tersedia dari Yahoo Finance.")
    st.stop()

# Reset index dan ambil kolom yang dibutuhkan
data.reset_index(inplace=True)
data = data[['Date', 'Open', 'High', 'Low', 'Close']].dropna()

# Hitung fitur tambahan
data['Days'] = (data['Date'] - data['Date'].min()).dt.days
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA100'] = data['Close'].rolling(window=100).mean()
data['Change%'] = data['Close'].pct_change() * 100
data['Lag1'] = data['Close'].shift(1)

# Bersihkan data (NaN dan outlier)
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

# Format harga ke Rupiah
data['Close_Rp'] = data['Close'].apply(lambda x: f"Rp {x:,.2f}")

# Pilihan jenis grafik
chart_type = st.selectbox("Pilih Jenis Grafik:", ["Line Chart", "Candlestick"])

fig = go.Figure()
if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Candlestick'
    ))
else:
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))

fig.update_layout(
    title='Grafik Harga Saham TLKM',
    xaxis_title='Tanggal',
    yaxis_title='Harga (Rupiah)',
    template='plotly_white',
    xaxis_rangeslider_visible=False
)
st.plotly_chart(fig, use_container_width=True)

# Modeling dan Evaluasi
X = data[['Days', 'MA50', 'MA100', 'Change%', 'Lag1']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi seluruh data
y_pred_all = model.predict(X)
mape = mean_absolute_percentage_error(y, y_pred_all) * 100
r2 = r2_score(y, y_pred_all)

# Visualisasi Prediksi vs Aktual 
st.subheader("Visualisasi Harga Aktual vs Prediksi")
comparison_plot = pd.DataFrame({
    'Tanggal': data['Date'],
    'Harga Aktual': y,
    'Harga Prediksi': y_pred_all
}).sort_values('Tanggal')

fig_comp = go.Figure()
fig_comp.add_trace(go.Scatter(x=comparison_plot['Tanggal'], y=comparison_plot['Harga Aktual'],
                              mode='lines', name='Harga Aktual', line=dict(color='blue')))
fig_comp.add_trace(go.Scatter(x=comparison_plot['Tanggal'], y=comparison_plot['Harga Prediksi'],
                              mode='lines', name='Harga Prediksi', line=dict(color='orange', dash='dash')))
fig_comp.update_layout(title='Harga Aktual vs Harga Prediksi (Seluruh Data)',
                       xaxis_title='Tanggal', yaxis_title='Harga (Rupiah)',
                       template='plotly_white')
st.plotly_chart(fig_comp, use_container_width=True)

# Prediksi 7 hari ke depan
last_day = data['Date'].max()
last_day_number = data['Days'].max()
last_MA50 = data['MA50'].iloc[-1]
last_MA100 = data['MA100'].iloc[-1]
last_Change = data['Change%'].iloc[-1]
last_Lag = data['Close'].iloc[-1]

future_dates = pd.bdate_range(start=last_day + dt.timedelta(days=1), periods=7).date

preds = []
lag = last_Lag
prev_price = lag

for i in range(7):
    day = last_day_number + i + 1
    change = ((lag - prev_price) / prev_price) * 100 if i > 0 else last_Change

    input_data = pd.DataFrame({
        'Days': [day],
        'MA50': [last_MA50],
        'MA100': [last_MA100],
        'Change%': [change],
        'Lag1': [lag]
    })

    pred = model.predict(input_data)[0]
    preds.append(pred)
    prev_price = lag
    lag = pred

forecast = pd.DataFrame({
    'Tanggal': future_dates,
    'Harga_Prediksi_Close': [f"Rp {x:,.2f}" for x in preds]
})

st.subheader("Prediksi Harga TLKM 7 Hari ke Depan")
st.dataframe(forecast)

# Tabel Perbandingan Harga Aktual vs Prediksi (10 terakhir)
comp = comparison_plot.copy()
comp['Tanggal'] = pd.to_datetime(comp['Tanggal']).dt.strftime('%Y-%m-%d')
comp['Harga Aktual'] = comp['Harga Aktual'].apply(lambda x: f"Rp {x:,.2f}")
comp['Harga Prediksi'] = comp['Harga Prediksi'].apply(lambda x: f"Rp {x:,.2f}")
st.subheader("Perbandingan Harga Aktual vs Prediksi")
st.dataframe(comp.tail(10).reset_index(drop=True))