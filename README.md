# Aplikasi Prediksi Harga Mobil Bekas

Aplikasi ini digunakan untuk memprediksi harga mobil bekas berdasarkan beberapa fitur yang dimasukkan oleh pengguna. Aplikasi ini dibangun menggunakan Streamlit dan beberapa algoritma pembelajaran mesin untuk melakukan prediksi harga mobil.

## Fitur Aplikasi

1. **Deskripsi Aplikasi**: Menyediakan informasi tentang aplikasi dan atribut-atribut yang digunakan untuk prediksi.
2. **Prediksi Harga Mobil**: Membolehkan pengguna memasukkan fitur mobil dan memprediksi harganya menggunakan salah satu dari beberapa model yang tersedia.

## Algoritma yang Digunakan

Aplikasi ini menggunakan beberapa algoritma pembelajaran mesin untuk melakukan prediksi harga mobil, antara lain:

1. **K-Nearest Neighbors (KNN)**: Algoritma yang menggunakan tetangga terdekat untuk melakukan prediksi. Algoritma ini mencari sejumlah k tetangga terdekat dalam data pelatihan dan menggunakan nilai rata-rata dari harga mobil di tetangga tersebut untuk melakukan prediksi.
2. **Linear Regression**: Metode statistik untuk memodelkan hubungan antara variabel dependen dengan satu atau lebih variabel independen. Model ini mengasumsikan hubungan linear antara variabel-variabel tersebut.
3. **Random Forest**: Algoritma ensemble learning yang menggunakan beberapa pohon keputusan untuk melakukan prediksi. Setiap pohon dalam hutan acak memberikan prediksi, dan hasil akhirnya adalah rata-rata dari semua prediksi pohon.
4. **Decision Tree**: Model prediksi yang menggunakan struktur pohon keputusan. Model ini membagi dataset menjadi subset berdasarkan fitur-fitur input dan melakukan prediksi berdasarkan nilai rata-rata dari subset akhir.

## Cara Instalasi

### 1. Clone Repository

Clone repository ini ke lokal mesin Anda

### 2.Membuat virtual environment
python -m venv myenv

### 3.Mengaktifkan virtual environment
## Untuk Windows
myenv\Scripts\activate

## Untuk macOS/Linux
source myenv/bin/activate

### 4.Menginstal Dependensi
pip install -r requirements.txt

### 5.Menjalankan Aplikasi
streamlit run app.py

### 6.Buka browser
  Local URL: http://localhost:8501
  Network URL: http://sesuai ip masing masing :8501
