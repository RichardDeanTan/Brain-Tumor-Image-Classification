# 🧠 Brain Tumor Image Classification

Proyek ini adalah implementasi aplikasi web deep learning untuk mengklasifikasikan gambar MRI otak ke dalam empat kategori berbeda: Glioma, Meningioma, Tanpa Tumor, dan Pituitary. Aplikasi ini memanfaatkan dua model canggih, EfficientNet B2 dan EfficientViT B1, serta menyediakan antarmuka interaktif bagi pengguna untuk mendapatkan prediksi secara instan.

## 📂 Project Structure

- `.streamlit/config.toml` — Konfigurasi Streamlit (auto darkmode).
- `model/` — Folder berisi model yang telah di save.
- `samples/` — DBerisi contoh gambar MRI untuk setiap kelas.
- `.gitignore` — File untuk mengabaikan folder atau file tertentu saat push ke Git.
- `Brain Tumor (EfficientNetB2).ipynb` — Notebook proses training dan fine-tuning model `EfficientNet B2`.
- `Brain Tumor (EfficientVitB1).ipynb` — Notebook proses training dan fine-tuning model `EfficientViT B1`.
- `app.py` — Aplikasi Streamlit utama yang di-deploy ke cloud (menggunakan best model `EfficientVitB1`).
- `full_app.py` — Aplikasi Streamlit untuk penggunaan lokal, terdapat opsi model `EfficientNetB2` dan `EfficientViTB1`.
- `reconstructing.py` — Python script untuk merapikan data sebelum proses training.
- `requirements.txt` — Daftar dependensi Python yang diperlukan untuk menjalankan proyek.

## 🚀 Cara Run Aplikasi

### 🔹 1. Jalankan Secara Lokal
### Clone Repository
```bash
git clone https://github.com/RichardDeanTan/Brain-Tumor-Image-Classification
cd Brain-Tumor-Image-Classification
```
### Install Dependensi
```bash
pip install -r requirements.txt
```
### Jalankan Aplikasi Streamlit
```bash
streamlit run full_app.py
```

### 🔹 2. Jalankan Secara Online (Tidak Perlu Install)
Klik link berikut untuk langsung membuka aplikasi web:
#### 👉 [Streamlit - Brain Tumor Image Classification](https://brain-tumor-image-classification-richardtanjaya.streamlit.app/)

## 💡 Fitur
- ✅ **Multi-Class Classification** — Mengklasifikasikan pindaian MRI otak ke dalam empat kategori: Glioma, Meningioma, Pituitary, dan Tanpa Tumor (No Tumor).
- ✅ **Dual Model Architecture** — Menggunakan dua arsitektur: CNN (`EfficientNet B2`) dan Vision Transformer (`EfficientViT B1`) untuk tugas klasifikasi.
- ✅ **High Accuracy Prediction** — Model `EfficientViT B1` yang digunakan dalam versi deployment mencapai akurasi hingga 98.05% pada test set.
- ✅ **Interactive Interface** — Pengguna dapat mengunggah gambar MRI (.jpg, .png, .jpeg) atau memilih dari contoh gambar yang tersedia untuk prediksi instan.
- ✅ **Confidence Score Visualization** — Menampilkan grafik batang (bar chart) dengan skor kepercayaan untuk setiap kelas menggunakan Plotly.
- ✅ **Model Performance Comparison** — Sidebar menampilkan perbandingan akurasi antara kedua model yang digunakan.
- ✅ **Informative Content** — Terdapat deskripsi yang jelas dan singkat untuk setiap jenis tumor otak agar pengguna memahami hasil prediksi.

## ⚙️ Tech Stack
- **Deep Learning Models** ~ TensorFlow/Keras, PyTorch
- **Arsitektur Model** ~ timm (`EfficientViT B1`)
- **Web Framework** ~ Streamlit
- **Manipulasi & Visualisasi Data** ~ NumPy, Plotly
- **Deployment** ~ Streamlit Cloud

## 🧠 Model Details
- **Model 1 (CNN)**
    - **Arsitektur:** `EfficientNetB2`
    - **Akurasi Test:** 89.65%
- **Model 2 (Vision Transformer)**
    - **Arsitektur:** `EfficientViT B1`
    - **Akurasi Test:** 98.05%
- **Tugas:** Klasifikasi Gambar Multi-Kelas
- **Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes)

## ⭐ Deployment
Aplikasi ini di-deploy menggunakan:
- Streamlit Cloud
- GitHub

## 👨‍💻 Pembuat
Richard Dean Tanjaya

## 📝 License
Proyek ini bersifat open-source dan bebas digunakan untuk keperluan edukasi dan penelitian.