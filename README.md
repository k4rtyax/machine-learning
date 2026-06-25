# Sistem Deteksi Intrusi Jaringan (IDS)

Implementasi Machine Learning untuk klasifikasi serangan siber menggunakan dataset CIC-IDS-2017.

**Live Demo:** [machine-learning-ta.streamlit.app](https://machine-learning-ta.streamlit.app)

## Model
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| XGBoost | 99.87% | 1.0000 |
| Ensemble (RF+XGB+ANN) | 99.87% | 0.9999 |
| Random Forest | 99.84% | 0.9999 |
| ANN | 97.94% | 0.9988 |

## Dataset
[CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) — Canadian Institute for Cybersecurity, ~2.8 juta sampel, 52 fitur numerik, 14 jenis serangan.

## Struktur
```
machine-learning/
├── app.py              # Streamlit dashboard
├── models/             # Model hasil training (.pkl, .h5)
├── notebooks/          # Notebook training & EDA
└── docs/               # Laporan dan dokumentasi
```
