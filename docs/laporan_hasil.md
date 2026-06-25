# Laporan Hasil Penelitian
## Implementasi Sistem Deteksi Intrusi Jaringan Menggunakan Machine Learning
**Kelompok 6 — Universitas Dian Nuswantoro**

---

## Informasi Dataset

| Item | Detail |
|---|---|
| Dataset | CIC-IDS-2017 (Canadian Institute for Cybersecurity) |
| File | `cicids2017_cleaned.csv` (~685 MB) |
| Total sampel (training) | 300.000 (subsample stratified) |
| Split | 80:20 stratified train_test_split |
| Fitur | 52 fitur numerik (diekstrak CICFlowMeter) |
| Label | 0 = Normal Traffic, 1 = Attack |
| Preprocessing | Drop inf/NaN, drop duplikat, StandardScaler |

---

## Hasil Evaluasi Model

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** ⭐ | **0.9987** | 0.9938 | **0.9987** | **0.9963** | **1.0000** |
| Ensemble (RF+XGB+ANN) | 0.9987 | 0.9937 | 0.9986 | 0.9962 | 0.9999 |
| Decision Tree | 0.9986 | 0.9943 | 0.9972 | 0.9958 | 0.9984 |
| Random Forest | 0.9984 | **0.9943** | 0.9963 | 0.9953 | 0.9999 |
| ANN | 0.9794 | 0.8951 | 0.9948 | 0.9423 | 0.9988 |
| Naive Bayes | 0.7754 | 0.4268 | 0.9601 | 0.5909 | 0.9225 |

---

## Model Terbaik: XGBoost

| Metrik | Nilai |
|---|---|
| Accuracy | 0.9987 (99.87%) |
| Precision | 0.9938 |
| Recall | 0.9987 |
| F1-Score | 0.9963 |
| ROC-AUC | 1.0000 |

---

## Analisis Per Model

### XGBoost ⭐ (Terbaik)
- Accuracy tertinggi: **99.87%**
- ROC-AUC sempurna: **1.0000** — model mampu memisahkan Normal vs Attack secara sempurna
- Recall 99.87% artinya hampir tidak ada serangan yang lolos (false negative sangat rendah)
- Cocok untuk produksi karena seimbang antara precision dan recall

### Ensemble (RF + XGB + ANN)
- Performa hampir identik dengan XGBoost (0.9987 vs 0.9987)
- Tidak memberikan peningkatan signifikan dibanding XGBoost saja
- Biaya komputasi lebih tinggi karena menjalankan 3 model

### Random Forest
- Accuracy 99.84% — sangat kompetitif
- Lebih mudah diinterpretasi melalui Feature Importance
- Ukuran model lebih besar di disk

### Decision Tree
- Accuracy 99.86% — mengejutkan, sangat baik
- Model paling ringan dan paling cepat saat inference
- Mudah divisualisasi dan dijelaskan ke stakeholder non-teknis

### ANN (Deep Learning)
- Accuracy 97.94% — paling rendah di antara tree-based models
- Precision rendah (0.8951) → lebih banyak false positive dibanding yang lain
- Recall tinggi (0.9948) → tetap bagus menangkap serangan
- Butuh lebih banyak data dan tuning untuk optimal

### Naive Bayes
- Accuracy 77.54% — jauh di bawah model lain
- Precision sangat rendah (0.4268) → banyak false alarm
- Tidak cocok untuk dataset ini karena asumsi independence antar fitur tidak terpenuhi

---

## Kesimpulan

1. **XGBoost** adalah model terbaik untuk dataset CIC-IDS-2017 dengan accuracy 99.87% dan ROC-AUC sempurna 1.0000
2. Semua model tree-based (XGBoost, RF, DT, Ensemble) memberikan performa sangat tinggi (>99.8%)
3. **Model yang direkomendasikan untuk deployment: XGBoost** — performa terbaik dengan ukuran model yang wajar
4. ANN memerlukan lebih banyak data dan hyperparameter tuning untuk bersaing dengan model tree-based
5. Naive Bayes tidak sesuai untuk masalah klasifikasi trafik jaringan ini

---

## File Model yang Dihasilkan

| File | Ukuran | Keterangan |
|---|---|---|
| `rf_model.pkl` | — | Random Forest (sklearn) |
| `xgb_model.pkl` | — | XGBoost |
| `ann_model.h5` | — | ANN (TensorFlow/Keras) |
| `scaler.pkl` | — | StandardScaler (preprocessing) |
| `feature_columns.pkl` | — | Daftar 52 nama fitur |

Lokasi: `models/`

---

## Konfigurasi Training

```python
# Random Forest
RandomForestClassifier(n_estimators=200, max_depth=20,
                       class_weight='balanced_subsample', random_state=42)

# XGBoost
XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
              subsample=0.8, colsample_bytree=0.8,
              scale_pos_weight=n_neg/n_pos, tree_method='hist')

# ANN
Sequential: Dense(128) → BN → Dropout(0.3) → Dense(64) → Dropout(0.2)
          → Dense(32) → Dropout(0.1) → Dense(1, sigmoid)
Optimizer: Adam | Loss: binary_crossentropy | Epochs: 30 | Batch: 2048
```

---

*Dibuat otomatis — 25 Juni 2026*
