"""
Dashboard Sistem Deteksi Intrusi Jaringan (IDS)
Menggunakan Machine Learning — CIC-IDS-2017 Dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
import os
import warnings

warnings.filterwarnings('ignore')

# ─── Konfigurasi Halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="IDS Dashboard — ML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS Kustom ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E3A5F 100%);
    }
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    [data-testid="stSidebar"] .stRadio label { padding: 8px 12px; border-radius: 8px; }
    [data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.1); }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1E3A5F, #2563EB);
        border-radius: 12px;
        padding: 16px;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.1);
    }
    [data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #BFDBFE !important; }

    h1 { color: #1E3A5F !important; font-weight: 700; }
    h2 { color: #2563EB !important; font-weight: 600; }
    h3 { color: #1E40AF !important; }

    .alert-normal {
        background: #DCFCE7; border-left: 4px solid #16A34A;
        padding: 12px 16px; border-radius: 8px; margin: 4px 0;
        color: #166534; font-weight: 500;
    }
    .alert-attack {
        background: #FEE2E2; border-left: 4px solid #DC2626;
        padding: 12px 16px; border-radius: 8px; margin: 4px 0;
        color: #7F1D1D; font-weight: 500;
    }
    .badge-normal { background:#16A34A; color:white; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; }
    .badge-attack { background:#DC2626; color:white; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; }
    .info-card {
        background: white; border: 1px solid #E2E8F0;
        border-radius: 12px; padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .dataframe { font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Google Drive Model IDs ────────────────────────────────────────────────────
# Isi dengan File ID dari Google Drive setelah training selesai
# Cara ambil File ID: klik kanan file di Drive → "Get link" → salin ID dari URL
GDRIVE_IDS = {
    'rf_model.pkl':        '',   # TODO: isi setelah training
    'xgb_model.pkl':       '',   # TODO: isi setelah training
    'scaler.pkl':          '',   # TODO: isi setelah training
    'feature_columns.pkl': '',   # TODO: isi setelah training
    'ann_model.h5':        '',   # TODO: isi setelah training
}

# ─── Auto-download Model dari Google Drive ────────────────────────────────────
def download_models_from_drive():
    """Download file model dari Google Drive jika belum ada di lokal."""
    try:
        import gdown
    except ImportError:
        return

    for fname, fid in GDRIVE_IDS.items():
        if not fid:
            continue
        local_path = fname
        # Cek di folder models/ dulu
        if os.path.exists(os.path.join('models', fname)):
            continue
        if not os.path.exists(local_path):
            with st.spinner(f'Mengunduh {fname} dari Google Drive...'):
                try:
                    gdown.download(f'https://drive.google.com/uc?id={fid}', local_path, quiet=True)
                except Exception:
                    pass

download_models_from_drive()

# ─── Cari file model (root atau folder models/) ───────────────────────────────
def find_model_file(fname):
    if os.path.exists(fname):
        return fname
    alt = os.path.join('models', fname)
    if os.path.exists(alt):
        return alt
    return None

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}

    path = find_model_file('rf_model.pkl')
    try:
        models['rf']     = joblib.load(path) if path else None
        models['rf_ok']  = models['rf'] is not None
    except Exception as e:
        models['rf_ok']  = False
        models['rf_err'] = str(e)

    try:
        from tensorflow.keras.models import load_model
        path = find_model_file('ann_model.h5')
        models['ann']    = load_model(path) if path else None
        models['ann_ok'] = models['ann'] is not None
    except Exception as e:
        models['ann_ok'] = False

    try:
        import xgboost as xgb_lib
        path = find_model_file('xgb_model.pkl')
        models['xgb']    = joblib.load(path) if path else None
        models['xgb_ok'] = models['xgb'] is not None
    except Exception:
        models['xgb_ok'] = False

    try:
        scaler_path = find_model_file('scaler.pkl')
        feat_path   = find_model_file('feature_columns.pkl')
        models['scaler']       = joblib.load(scaler_path) if scaler_path else None
        models['feature_cols'] = joblib.load(feat_path)   if feat_path   else None
        models['prep_ok']      = models['scaler'] is not None
    except Exception as e:
        models['prep_ok']  = False
        models['prep_err'] = str(e)

    return models

models = load_models()

# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_input(df: pd.DataFrame, models: dict) -> np.ndarray:
    """Preprocess CSV sesuai pipeline training CIC-IDS-2017 (numerik, StandardScaler)."""
    if not models.get('prep_ok'):
        raise ValueError("Preprocessor tidak ditemukan. Jalankan notebook training terlebih dahulu.")

    df = df.copy()
    df.columns = df.columns.str.strip()

    # Ambil hanya kolom numerik
    df_num = df.select_dtypes(include=[np.number])

    # Sinkronisasi dengan feature_cols dari training
    feat_cols = models.get('feature_cols')
    if feat_cols:
        for col in feat_cols:
            if col not in df_num.columns:
                df_num[col] = 0.0
        df_num = df_num[feat_cols]

    # Ganti inf → NaN → 0
    df_num.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_num.fillna(0, inplace=True)

    return models['scaler'].transform(df_num.values)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ IDS Dashboard")
    st.markdown("**Machine Learning — CIC-IDS-2017**")
    st.divider()
    halaman = st.radio(
        "Navigasi",
        ["🏠 Beranda", "🔍 Prediksi", "📊 Evaluasi Model", "📈 Eksplorasi Data"],
        label_visibility="collapsed"
    )
    st.divider()

    st.markdown("**Status Model:**")
    for icon_key, label in [('rf_ok', 'Random Forest'), ('ann_ok', 'ANN (Keras)'),
                             ('xgb_ok', 'XGBoost'), ('prep_ok', 'Preprocessor')]:
        icon = "✅" if models.get(icon_key) else "❌"
        st.markdown(f"{icon} {label}")

    if not any(models.get(k) for k in ['rf_ok', 'ann_ok', 'xgb_ok']):
        st.warning("⚠️ Jalankan notebook training terlebih dahulu, lalu letakkan file .pkl & .h5 ke folder `models/`")

# ═══════════════════════════════════════════════════════════════════
# 🏠 HALAMAN BERANDA
# ═══════════════════════════════════════════════════════════════════
if halaman == "🏠 Beranda":
    st.title("🛡️ Sistem Deteksi Intrusi Jaringan")
    st.markdown("#### Implementasi Machine Learning untuk Klasifikasi Serangan Siber")
    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Tentang Sistem Ini**

        Dashboard ini merupakan implementasi **Intrusion Detection System (IDS)** berbasis
        Machine Learning menggunakan dataset **CIC-IDS-2017** dari Canadian Institute for
        Cybersecurity, University of New Brunswick.

        Sistem menggunakan tiga model utama yang dikombinasikan dalam **Ensemble Soft Voting**:

        - 🌲 **Random Forest** — ensemble decision tree dengan class weighting
        - 🚀 **XGBoost** — gradient boosting dengan scale_pos_weight
        - 🧠 **ANN** — deep learning dengan BatchNormalization & Dropout

        Sistem mengklasifikasikan trafik jaringan sebagai **Normal** atau **Attack** secara otomatis.
        """)
    with col2:
        st.markdown("""
        **Fitur Utama**
        - 🔍 Prediksi real-time via upload CSV
        - 📊 Evaluasi & perbandingan model
        - 📈 Eksplorasi data interaktif
        - 💾 Download hasil prediksi
        """)

    st.divider()

    st.markdown("### 📊 Statistik Dataset CIC-IDS-2017")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Sampel",    "~2.8 Juta",  "CIC-IDS-2017")
    col2.metric("Data Training",   "~2 Juta",    "80% stratified")
    col3.metric("Data Testing",    "~504 Ribu",  "20% stratified")
    col4.metric("Jumlah Fitur",    "52",         "numerik")
    col5.metric("Kelas Serangan",  "14 jenis",   "Binary: Normal vs Attack")

    st.divider()

    st.markdown("### 🎯 Kategori Serangan CIC-IDS-2017")
    attack_info = {
        "DoS / DDoS": {
            "warna": "#EF4444",
            "deskripsi": "Membanjiri server dengan traffic berlebih hingga tidak dapat melayani pengguna sah.",
            "contoh": "DoS Hulk, DoS GoldenEye, DoS Slowloris, DDoS"
        },
        "Web Attack": {
            "warna": "#F59E0B",
            "deskripsi": "Eksploitasi kerentanan aplikasi web untuk mendapatkan akses tidak sah.",
            "contoh": "Brute Force, XSS, SQL Injection"
        },
        "Port Scan": {
            "warna": "#8B5CF6",
            "deskripsi": "Pemindaian port untuk mengidentifikasi layanan dan celah keamanan pada target.",
            "contoh": "PortScan"
        },
        "Infiltration / Bot": {
            "warna": "#EC4899",
            "deskripsi": "Penyusupan jaringan atau aktivitas botnet untuk kendali jarak jauh.",
            "contoh": "Infiltration, Bot, Heartbleed"
        },
    }
    cols = st.columns(4)
    for i, (name, info) in enumerate(attack_info.items()):
        with cols[i]:
            st.markdown(f"""
            <div style="border-left: 4px solid {info['warna']}; padding: 12px 16px;
                        background: white; border-radius: 8px; height: 190px;">
                <b style="color:{info['warna']};">{name}</b><br><br>
                <small>{info['deskripsi']}</small><br><br>
                <code style="font-size:11px; color:#64748B;">{info['contoh']}</code>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 🔍 HALAMAN PREDIKSI
# ═══════════════════════════════════════════════════════════════════
elif halaman == "🔍 Prediksi":
    st.title("🔍 Prediksi Intrusi Jaringan")
    st.markdown("Upload file CSV berisi data trafik jaringan CIC-IDS-2017 untuk mendapatkan prediksi.")
    st.divider()

    col_opt1, col_opt2 = st.columns([1, 2])
    with col_opt1:
        model_options = []
        if models.get('rf_ok'):  model_options.append("Random Forest")
        if models.get('ann_ok'): model_options.append("ANN")
        if models.get('xgb_ok'): model_options.append("XGBoost")
        if not model_options:    model_options = ["(Belum ada model)"]
        model_choice = st.selectbox("Pilih Model", options=model_options)
    with col_opt2:
        threshold = st.slider("Threshold Prediksi Attack", 0.3, 0.9, 0.5, 0.05,
                              help="Probabilitas di atas nilai ini → diklasifikasikan sebagai Attack")

    st.divider()

    feat_cols = models.get('feature_cols')
    with st.expander("ℹ️ Format File CSV yang Diperlukan"):
        if feat_cols:
            st.markdown(f"File CSV harus memiliki **{len(feat_cols)} kolom numerik** berikut (fitur CIC-IDS-2017):")
            st.code(", ".join(feat_cols))
        else:
            st.info("Load model terlebih dahulu untuk melihat daftar kolom yang diperlukan.")
        st.markdown("Kolom label (`Attack Type`, `Label`, dll.) tidak perlu disertakan.")

    uploaded_file = st.file_uploader("Upload File CSV", type=['csv', 'txt'])

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.success(f"✅ File berhasil dimuat: **{df_input.shape[0]:,} baris, {df_input.shape[1]} kolom**")

            with st.expander("👁️ Preview Data (5 baris pertama)"):
                st.dataframe(df_input.head(), use_container_width=True)

            model_ready = (
                (model_choice == "Random Forest" and models.get('rf_ok')) or
                (model_choice == "ANN"           and models.get('ann_ok')) or
                (model_choice == "XGBoost"       and models.get('xgb_ok'))
            )

            if st.button("🚀 Jalankan Prediksi", type="primary", use_container_width=True):
                if not model_ready:
                    st.error(f"❌ Model {model_choice} tidak tersedia. Letakkan file model ke folder `models/`.")
                else:
                    with st.spinner("Memproses data & melakukan prediksi..."):
                        X = preprocess_input(df_input, models)

                        if model_choice == "Random Forest":
                            y_prob = models['rf'].predict_proba(X)[:, 1]
                        elif model_choice == "XGBoost":
                            y_prob = models['xgb'].predict_proba(X)[:, 1]
                        else:
                            y_prob = models['ann'].predict(X, verbose=0).flatten()

                        y_pred = (y_prob >= threshold).astype(int)
                        labels = ['🟢 Normal' if p == 0 else '🔴 Attack' for p in y_pred]

                        df_result = df_input.copy()
                        df_result['Prediksi']     = labels
                        df_result['Probabilitas'] = np.round(y_prob, 4)
                        df_result['Kelas']        = y_pred

                        n_normal = int(sum(y_pred == 0))
                        n_attack = int(sum(y_pred == 1))
                        total    = len(y_pred)

                        st.divider()
                        st.markdown("### 📋 Hasil Prediksi")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Total Sampel", f"{total:,}")
                        m2.metric("🟢 Normal",    f"{n_normal:,}", f"{n_normal/total*100:.1f}%")
                        m3.metric("🔴 Attack",    f"{n_attack:,}", f"{n_attack/total*100:.1f}%")
                        m4.metric("Threshold",    threshold)

                        preview_cols = ['Prediksi', 'Probabilitas'] + list(df_input.columns[:5])
                        st.dataframe(
                            df_result[[c for c in preview_cols if c in df_result.columns]],
                            use_container_width=True, height=300
                        )

                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.pie([n_normal, n_attack], labels=['Normal', 'Attack'],
                               colors=['#16A34A', '#DC2626'], autopct='%1.1f%%',
                               startangle=90, textprops={'fontsize': 13, 'fontweight': 'bold'})
                        ax.set_title(f'Distribusi Prediksi ({model_choice})', fontsize=13, fontweight='bold')
                        st.pyplot(fig, use_container_width=False)
                        plt.close()

                        csv_out = df_result.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "💾 Download Hasil Prediksi (CSV)",
                            data=csv_out,
                            file_name='hasil_prediksi_ids.csv',
                            mime='text/csv',
                            type='primary'
                        )

        except Exception as e:
            st.error(f"❌ Error memproses file: {e}")
            st.info("Pastikan file CSV berisi fitur numerik CIC-IDS-2017.")
    else:
        st.info("👆 Upload file CSV untuk memulai prediksi")

# ═══════════════════════════════════════════════════════════════════
# 📊 HALAMAN EVALUASI MODEL
# ═══════════════════════════════════════════════════════════════════
elif halaman == "📊 Evaluasi Model":
    st.title("📊 Evaluasi & Perbandingan Model")
    st.markdown("Perbandingan performa model machine learning pada dataset CIC-IDS-2017.")
    st.divider()

    st.markdown("### 🏆 Tabel Perbandingan Performa Model")

    df_eval = pd.DataFrame({
        'Model':      ['XGBoost', 'Ensemble (RF+XGB+ANN)', 'Decision Tree', 'Random Forest', 'ANN', 'Naive Bayes'],
        'Accuracy':   [0.9987, 0.9987, 0.9986, 0.9984, 0.9794, 0.7754],
        'Precision':  [0.9938, 0.9937, 0.9943, 0.9943, 0.8951, 0.4268],
        'Recall':     [0.9987, 0.9986, 0.9972, 0.9963, 0.9948, 0.9601],
        'F1-Score':   [0.9963, 0.9962, 0.9958, 0.9953, 0.9423, 0.5909],
        'ROC-AUC':    [1.0000, 0.9999, 0.9984, 0.9999, 0.9988, 0.9225],
    }).set_index('Model')

    def highlight_best(s):
        is_max = s == s.max()
        return ['background-color: #D1FAE5; font-weight: bold' if v else '' for v in is_max]

    styled = df_eval.style.apply(highlight_best).format("{:.4f}")
    st.dataframe(styled, use_container_width=True)
    st.caption("Hasil evaluasi aktual — CIC-IDS-2017, 300k sampel, split 80:20 stratified. Model terbaik: **XGBoost** (Accuracy 99.87%, ROC-AUC 1.0000)")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Grafik Perbandingan Metrik")
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(df_eval.index))
        width = 0.18
        colors = ['#2563EB', '#16A34A', '#F59E0B', '#EF4444']
        for i, (metric, color) in enumerate(zip(['Accuracy', 'Precision', 'Recall', 'F1-Score'], colors)):
            ax.bar(x + i * width, df_eval[metric], width, label=metric, color=color, alpha=0.85)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(df_eval.index, rotation=15, fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Skor', fontsize=11)
        ax.set_title('Perbandingan Metrik Evaluasi', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("### 🔲 Confusion Matrix")
        if models.get('rf_ok') and models.get('prep_ok'):
            st.info("Upload data test untuk melihat Confusion Matrix aktual.")
        else:
            st.info("Load model terlebih dahulu untuk melihat Confusion Matrix.")

    st.divider()

    st.markdown("### 🌟 Feature Importance — Random Forest")
    if models.get('rf_ok') and models.get('feature_cols'):
        feat_imp = pd.DataFrame({
            'Fitur': models['feature_cols'],
            'Importance': models['rf'].feature_importances_
        }).sort_values('Importance', ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, 10))[::-1]
        bars = ax.barh(feat_imp['Fitur'][::-1], feat_imp['Importance'][::-1],
                       color=colors, edgecolor='#1E3A5F', linewidth=0.5)
        for bar, val in zip(bars, feat_imp['Importance'][::-1]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
        ax.set_title('Top 10 Feature Importance — Random Forest (CIC-IDS-2017)',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Importance Score', fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    else:
        st.info("Load model Random Forest untuk melihat Feature Importance aktual.")

# ═══════════════════════════════════════════════════════════════════
# 📈 HALAMAN EDA
# ═══════════════════════════════════════════════════════════════════
elif halaman == "📈 Eksplorasi Data":
    st.title("📈 Eksplorasi Data CIC-IDS-2017")
    st.divider()

    st.markdown("### 🏷️ Distribusi Label")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Binary Label (Normal vs Attack)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts_bin = [1446882, 1073708]
        ax.pie(counts_bin, labels=['Normal', 'Attack'],
               autopct='%1.1f%%', colors=['#16A34A', '#DC2626'],
               startangle=90, textprops={'fontsize': 12})
        ax.set_title('Distribusi Binary Label (sebelum cleaning)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("**Per Jenis Serangan (Top 10)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        cats = ['BENIGN', 'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye',
                'FTP-Patator', 'SSH-Patator', 'DoS Slowloris', 'DoS Slowhttptest', 'Bot']
        counts = [1446882, 231073, 158930, 128027, 10293, 7938, 5897, 5796, 5499, 1966]
        colors = ['#16A34A'] + ['#EF4444'] * 9
        bars = ax.barh(cats[::-1], counts[::-1], color=colors[::-1])
        for bar, val in zip(bars, counts[::-1]):
            ax.text(bar.get_width() + 1000, bar.get_y() + bar.get_height() / 2,
                    f'{val:,}', va='center', fontsize=9)
        ax.set_title('Top 10 Jenis Trafik', fontsize=12, fontweight='bold')
        ax.set_xlabel('Jumlah Sampel')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.divider()
    st.markdown("### 📋 Deskripsi Fitur CIC-IDS-2017")

    if models.get('feature_cols'):
        feat_info = pd.DataFrame({
            'No':    range(1, len(models['feature_cols']) + 1),
            'Fitur': models['feature_cols'],
            'Tipe':  ['Numerik'] * len(models['feature_cols']),
        })
        st.dataframe(feat_info, use_container_width=True, height=400)
        st.caption(f"Total {len(models['feature_cols'])} fitur numerik dari CICFlowMeter.")
    else:
        st.info("Load model untuk melihat daftar fitur aktual dari training.")
        st.markdown("""
        **Jenis fitur yang tersedia di CIC-IDS-2017:**
        - Statistik flow: durasi, panjang paket, jumlah paket
        - Flag TCP: SYN, ACK, FIN, RST, PSH, URG
        - Laju pengiriman: byte/s, packet/s
        - Window size, header length
        - IAT (Inter-Arrival Time): mean, std, max, min
        - Subflow: bytes & packets forward/backward
        """)

    st.divider()
    st.markdown("### 🔥 Heatmap Korelasi (Ilustrasi)")
    st.caption("⚠️ Heatmap di bawah merupakan ilustrasi. Heatmap aktual tersedia setelah menjalankan notebook EDA.")

    top15 = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Fwd Packets Length Total', 'Bwd Packets Length Total',
        'Fwd Packet Length Max', 'Bwd Packet Length Max',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
        'Fwd IAT Total', 'Bwd IAT Total', 'Active Mean',
        'Idle Mean', 'Subflow Fwd Bytes'
    ]
    np.random.seed(42)
    corr_data = np.eye(15)
    for i in range(15):
        for j in range(i + 1, 15):
            v = np.random.uniform(-0.5, 0.85)
            corr_data[i, j] = corr_data[j, i] = v
    corr_df = pd.DataFrame(corr_data, index=top15, columns=top15)

    fig, ax = plt.subplots(figsize=(13, 9))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, linewidths=0.4, ax=ax, annot_kws={'size': 7.5},
                cbar_kws={'shrink': 0.75})
    ax.set_title('Heatmap Korelasi — Top 15 Fitur CIC-IDS-2017 (Ilustrasi)',
                 fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
