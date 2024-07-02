import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
try:
    knn = joblib.load('models/knn_model.pkl')
    linear_reg = joblib.load('models/linear_reg_model.pkl')
    rf = joblib.load('models/rf_model.pkl')
    dt = joblib.load('models/dt_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoders = {}
    for column in ['model', 'transmisi', 'bahan_bakar']:
        label_encoders[column] = joblib.load(f'models/label_encoder_{column}.pkl')
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Streamlit app
st.title("Used Car Price Prediction")

st.sidebar.header('Car Specifications')
def user_input_features():
    model = st.sidebar.selectbox('Model', label_encoders['model'].classes_)
    tahun = st.sidebar.slider('Year', 1870, 2022, 2020)
    transmisi = st.sidebar.selectbox('Transmission', label_encoders['transmisi'].classes_)
    jarak_tempuh = st.sidebar.number_input('Mileage', 0, 10000000, 50000)
    bahan_bakar = st.sidebar.selectbox('Fuel Type', label_encoders['bahan_bakar'].classes_)
    pajak = st.sidebar.number_input('Tax', 0, 5000000, 1500000)
    mpg = st.sidebar.number_input('MPG', 0, 100, 25)
    ukuran_mesin = st.sidebar.number_input('Engine Size', 0.0, 10.0, 1.2)
    
    input_data = {'model': model,
                  'tahun': tahun,
                  'transmisi': transmisi,
                  'jarak_tempuh': jarak_tempuh,
                  'bahan_bakar': bahan_bakar,
                  'pajak': pajak,
                  'mpg': mpg,
                  'ukuran_mesin': ukuran_mesin}
    features = pd.DataFrame(input_data, index=[0])
    return features

input_df = user_input_features()

# Mengonversi fitur kategorikal menjadi numerik menggunakan encoder yang telah dilatih
for column in ['model', 'transmisi', 'bahan_bakar']:
    input_df[column] = label_encoders[column].transform(input_df[column])

# Memilih hanya fitur yang digunakan dalam model
input_X = input_df[['tahun', 'jarak_tempuh', 'ukuran_mesin']]

# Standarisasi fitur numerik menggunakan scaler yang telah dilatih
input_X = scaler.transform(input_X)

# Prediksi menggunakan semua model yang telah dilatih
pred_knn = knn.predict(input_X)[0]
pred_lr = linear_reg.predict(input_X)[0]
pred_rf = rf.predict(input_X)[0]
pred_dt = dt.predict(input_X)[0]

# Menampilkan hasil prediksi menggunakan pop-up
st.subheader('Predicted Price')
if st.button('Predict'):
    st.write(f'Prediksi harga menggunakan KNN: {pred_knn:,.2f} Rupiah')
    st.write(f'Prediksi harga menggunakan Regresi Linier: {pred_lr:,.2f} Rupiah')
    st.write(f'Prediksi harga menggunakan Random Forest: {pred_rf:,.2f} Rupiah')
    st.write(f'Prediksi harga menggunakan Decision Tree: {pred_dt:,.2f} Rupiah')
