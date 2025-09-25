# Rakamin Final Project - Batch 56
## Anggota
- 👨‍💼 Project Manager         → Dadin Tajudin
- 🛠️ Data Engineer           → Athariq Marsha Nugraha
- 🧑‍🔬 Data Scientist          → Nada Paradita
- 📊 Business & Data Analyst → Nida Febiana

# 🚀 Employee Churn Prediction - Rakamin Finpro DS56 Kelompok 2

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Modeling-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-GradientBoosting-green)
![CatBoost](https://img.shields.io/badge/CatBoost-Boosting-yellow)

---

## 📌 Tentang Proyek
Project ini merupakan Final Project dari **Rakamin Data Science Bootcamp Batch 56 (Kelompok 2)**.  
Tujuan proyek ini adalah **memprediksi churn (resign) karyawan** menggunakan beberapa algoritma machine learning dan membuat **Rapid Web Apps prototyping** yang interaktif.  

👉 Dengan model ini, perusahaan dapat:
- ✅ Mengidentifikasi potensi churn karyawan lebih awal.  
- ✅ Menjalankan simulasi pengurangan churn.  
- ✅ Mengestimasi potensi **penghematan biaya** akibat churn.  

---

## 🧰 Algoritma yang Digunakan
Model prediksi dibangun menggunakan **6 algoritma machine learning**:
- 🟦 Logistic Regression  
- 🌳 Decision Tree  
- 🧑‍🤝‍🧑 K-Nearest Neighbors (KNN)  
- 🌲 Random Forest  
- ⚡ XGBoost  
- 🐱 CatBoost  

Setiap model dilakukan **hyperparameter tuning** untuk hasil yang optimal.  

## 📊 Hasil & Output
### Model terbaik ditentukan dari evaluasi metrik (Recall, F2-score, ROC-AUC).
### Aplikasi web menyediakan:
- 🧑‍💼 Simulasi churn karyawan baru.
- 💸 Estimasi potensi **pengurangan biaya** akibat churn.
- 📈 Analisa data karyawan dalam bentuk visualisasi.
---

## Employee Churn Prediction App
Aplikasi ini adalah dashboard interaktif berbasis Streamlit untuk melakukan analisis data dan prediksi churn karyawan. Aplikasi ini mencakup fitur EDA (Exploratory Data Analysis), prediksi perorangan, prediksi batch, dan analisis penghematan biaya.

### 📁 Struktur Folder
```
project-folder/
├── app.py
├── eda_module.py
├── utils.py
├── employee_churn_prediction_updated.csv
├── requirements.txt
└── ...
```
---
### ⚙️ Instalasi
1. **Clone repositori atau salin file ke folder lokal**
2. **Aktifkan virtual environment (opsional tapi disarankan)**
3. **Install dependensi**

```bash
pip install -r requirements.txt
```
Jika tidak ada `requirements.txt`, install manual:
```bash
pip install streamlit pandas scikit-learn matplotlib seaborn
```
---

### 🚀 Menjalankan Aplikasi

```bash
streamlit run app.py
```
---
### 🧭 Navigasi Aplikasi
Gunakan sidebar di sebelah kiri untuk memilih halaman:

#### 1. 📊 Dashboard (EDA)
- Menampilkan ringkasan dataset
- Visualisasi missing values, distribusi target, fitur numerik & kategorikal
- Korelasi antar fitur numerik

#### 2. 👤 Prediksi Perorangan
- Masukkan data karyawan secara manual
- Dapatkan prediksi churn dan estimasi penghematan jika churn dicegah

#### 3. 📂 Prediksi Batch
- Upload file CSV berisi data karyawan
- Dapatkan hasil prediksi churn secara massal dan unduh hasilnya

#### 4. 💰 Analisis Penghematan Biaya
- Simulasikan penghematan biaya berdasarkan efektivitas intervensi dan biaya tindakan
- Visualisasi kurva net saving terhadap threshold

---

### 📌 Catatan
- Pastikan file dataset (`employee_churn_prediction_updated.csv`) tersedia di direktori yang sama dengan `app.py`, atau sesuaikan path-nya di sidebar.
- File `utils.py` dan `eda_module.py` harus tersedia karena berisi fungsi pendukung.
