# 🧠 Capstone Project Module 3 — Prediksi Customer Lifetime Value

## 📘 Deskripsi Proyek
Proyek ini merupakan tugas akhir Modul 3 dalam program Data Analyst, dengan tujuan membangun model machine learning untuk memprediksi **Customer Lifetime Value (CLV)** pada perusahaan **asuransi mobil**.  
Model dibuat untuk membantu perusahaan memahami nilai pelanggan dan menentukan strategi retensi yang lebih efektif.

---

## 🎯 Tujuan
- Memprediksi **Customer Lifetime Value** berdasarkan data pelanggan.  
- Menentukan **faktor-faktor yang paling berpengaruh** terhadap nilai CLV.  
- Membandingkan performa beberapa algoritma regresi untuk menentukan model terbaik.

---

## 🧩 Dataset
Dataset yang digunakan: **Customer Lifetime Value (Auto Insurance Dataset)**  
Beberapa kolom penting:
- `Customer Lifetime Value` — nilai total yang dihasilkan pelanggan.  
- `Monthly Premium Auto` — jumlah premi asuransi mobil per bulan.  
- `Income` — pendapatan pelanggan.  
- `Vehicle Class`, `Coverage`, `Education`, `Gender`, dll — atribut demografis dan polis pelanggan.

---

## ⚙️ Tahapan Pengerjaan
1. **Data Understanding** – Analisis kolom dan konteks bisnis.  
2. **Data Cleaning & Preprocessing** – Menangani missing value, encoding variabel kategorikal, dan scaling data numerik.  
3. **Modeling** – Menerapkan tiga model regresi:
   - Linear Regression  
   - Random Forest Regressor  
   - XGBoost Regressor  
4. **Evaluasi Model** – Menggunakan metrik:
   - MAE, RMSE, R², dan MAPE  
   Model terbaik ditentukan berdasarkan **nilai RMSE terkecil**.  
5. **Feature Importance** – Mengidentifikasi fitur-fitur paling berpengaruh terhadap CLV.  

---

## 🧮 Hasil
- **Model terbaik:** Random Forest Regressor  
- **Alasan:** Memiliki nilai RMSE terkecil dibandingkan model lainnya.  
- **Fitur paling berpengaruh:**  
  `Monthly Premium Auto`, `Income`, `Months Since Policy Inception`, dan `Total Claim Amount`.

---

## 📦 File yang Disertakan
| File | Deskripsi |
|------|------------|
| `Capstone Project Module 3.ipynb` | Notebook utama berisi seluruh proses analisis dan modeling |
| `model_capstone_project_modul3.pkl` | Model final yang disimpan menggunakan Pickle |
| `data_customer_lifetime_value.csv` | Dataset yang digunakan untuk pelatihan model |

---

## 🧾 Kesimpulan
Model Random Forest berhasil memberikan hasil terbaik dalam memprediksi nilai **Customer Lifetime Value**, dengan performa yang stabil dan error terkecil berdasarkan RMSE.  
Model ini dapat digunakan untuk memperkirakan nilai pelanggan baru dengan karakteristik serupa, sehingga membantu strategi bisnis dalam meningkatkan retensi pelanggan.

---

## 👤 Pembuat
**Nama:** Taufiq Bariz  
**Program:** Data Analyst Bootcamp — Purwadhika  
