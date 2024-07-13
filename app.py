import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import base64
import os

# Load dataset
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

data = load_data('data/mmm.csv')

# Encode categorical data
le_model = LabelEncoder()
data['model_encoded'] = le_model.fit_transform(data['model'])

# Pemilihan Fitur dan Target
X = data[['tahun', 'jarak_tempuh', 'pajak', 'mpg', 'ukuran_mesin', 'model_encoded']]
y = data['harga']

# Pembagian Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi Fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
def train_models():
    models = {
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100),
        'Decision Tree': DecisionTreeRegressor()
    }
    for model in models.values():
        model.fit(X_train_scaled, y_train)
    return models

models = train_models()
model_scores = {name: mean_squared_error(y_test, model.predict(X_test_scaled)) for name, model in models.items()}

# Function to add background image from local file
def add_bg_from_local(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as image:
            encoded_string = base64.b64encode(image.read())
        st.markdown(
            f"""
            <style>
            .background-img {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 300px;
                background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
                background-size: cover;
                background-repeat: no-repeat;
                z-index: -1;
            }}
            .card {{
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                background: none;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Add background image
add_bg_from_local('data/background.jpg')

# CSS for button style
st.markdown(
    """
    <style>
    .button {
        display: inline-block;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        text-align: center;
        text-decoration: none;
        outline: none;
        color: #fff;
        background-color: #4CAF50;
        border: none;
        border-radius: 15px;
        box-shadow: 0 9px #999;
    }
    .button:hover {background-color: #3e8e41}
    .button:active {
        background-color: #3e8e41;
        box-shadow: 0 5px #666;
        transform: translateY(4px);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.markdown('<div class="background-img"></div>', unsafe_allow_html=True)
st.title('Car Price Prediction')

# Display image below title
st.image('data/background.jpg', caption='Background Image', use_column_width=True)

# Sidebar menu using buttons
st.sidebar.header('Menu')
menu_selection = st.sidebar.radio("", ('Deskripsi Aplikasi', 'Prediksi Harga Mobil'))

if menu_selection == 'Deskripsi Aplikasi':
    st.header('Deskripsi Aplikasi')
    st.markdown("""
    <div class="card">
    Aplikasi ini digunakan untuk memprediksi harga mobil bekas berdasarkan beberapa fitur yang dimasukkan oleh pengguna.
    
    ### Penjelasan Atribut:
    - **tahun**: Tahun pembuatan mobil.
    - **jarak_tempuh**: Jarak tempuh mobil dalam kilometer.
    - **pajak**: Pajak mobil.
    - **mpg**: Efisiensi bahan bakar dalam mil per galon.
    - **ukuran_mesin**: Ukuran mesin dalam liter.
    
    ### Model Prediksi yang Digunakan:
    """, unsafe_allow_html=True)
    
    with st.expander("K-Nearest Neighbors (KNN)"):
        st.write("""
        KNN adalah algoritma yang menggunakan tetangga terdekat untuk melakukan prediksi. Untuk setiap titik data baru, algoritma ini akan mencari sejumlah k tetangga terdekat dalam data pelatihan dan menggunakan nilai rata-rata dari harga mobil di tetangga tersebut untuk melakukan prediksi.
        """)
    with st.expander("Linear Regression"):
        st.write("""
        Linear Regression adalah metode statistik untuk memodelkan hubungan antara variabel dependen dengan satu atau lebih variabel independen. Model ini mengasumsikan hubungan linear antara variabel-variabel tersebut.
        """)
    with st.expander("Random Forest"):
        st.write("""
        Random Forest adalah algoritma ensemble learning yang menggunakan beberapa pohon keputusan untuk melakukan prediksi. Setiap pohon dalam hutan acak memberikan prediksi, dan hasil akhir adalah rata-rata dari semua prediksi pohon.
        """)
    with st.expander("Decision Tree"):
        st.write("""
        Decision Tree adalah model prediksi yang menggunakan struktur pohon keputusan. Model ini membagi dataset menjadi subset berdasarkan fitur-fitur input, dan melakukan prediksi berdasarkan nilai rata-rata dari subset akhir.
        """)
    
    st.markdown("""
    </div>
    <div class="card">
    ## Evaluasi Model
    """, unsafe_allow_html=True)
    st.write(model_scores)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    ## Data Visualization
    """, unsafe_allow_html=True)
    fig, ax = plt.subplots()
    sns.scatterplot(x='tahun', y='harga', data=data, ax=ax)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

elif menu_selection == 'Prediksi Harga Mobil':
    st.header('Prediksi Harga Mobil')
    
    st.markdown("""
    <div class="card">
    ## Input Features
    """, unsafe_allow_html=True)

    # Input fields
    tahun = st.number_input('Year', min_value=1900, max_value=2024, value=2020)
    jarak_tempuh = st.number_input('Mileage', min_value=0, max_value=1000000, value=50000)
    pajak = st.number_input('Tax', min_value=0, max_value=5000000, value=200000)
    mpg = st.number_input('MPG', min_value=0, max_value=100, value=25)
    ukuran_mesin = st.number_input('Engine Size', min_value=0.0, max_value=10.0, value=1.5)
    model = st.selectbox('Model', le_model.classes_)

    # Encode model input
    model_encoded = le_model.transform([model])[0]

    # Model selection
    model_choice = st.selectbox('Choose Model', ['KNN', 'Linear Regression', 'Random Forest', 'Decision Tree'])

    st.markdown("</div>", unsafe_allow_html=True)

    # Predict button
    if st.button('Predict'):
        features = np.array([[tahun, jarak_tempuh, pajak, mpg, ukuran_mesin, model_encoded]])
        features_scaled = scaler.transform(features)
        if model_choice == 'KNN':
            prediction = models['KNN'].predict(features_scaled)
        elif model_choice == 'Linear Regression':
            prediction = models['Linear Regression'].predict(features_scaled)
        elif model_choice == 'Random Forest':
            prediction = models['Random Forest'].predict(features_scaled)
        else:
            prediction = models['Decision Tree'].predict(features_scaled)
        
        # Format the prediction in Rupiah
        formatted_price = f"Rp {prediction[0]:,.0f}".replace(',', '.')
        
        # Display the result below the Predict button
        st.markdown(f"""
        <div class="card">
        ## Predicted Price: {formatted_price}
        </div>
        """, unsafe_allow_html=True)
