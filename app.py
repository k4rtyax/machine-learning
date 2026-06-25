"""
Dashboard Sistem Deteksi Intrusi Jaringan (IDS)
Menggunakan Machine Learning -- CIC-IDS-2017 Dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# ── Konfigurasi Halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="IDS Dashboard -- ML",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Kustom ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Hapus header toolbar Streamlit */
    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }

    /* Kurangi padding bawaan Streamlit */
    .main .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1300px;
    }
    /* Kompres jarak antar elemen */
    div[data-testid="stVerticalBlock"] > div { gap: 0.3rem; }
    hr { margin: 0.4rem 0 !important; }
    h1 { margin-top: 0 !important; margin-bottom: 0.1rem !important; }
    p  { margin-bottom: 0.3rem !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        padding-top: 0.5rem !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label { color: #CBD5E1; }
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] strong { color: #F1F5F9 !important; }
    [data-testid="stSidebar"] .stRadio label {
        padding: 8px 12px; border-radius: 8px;
        transition: background 0.15s;
    }
    [data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.08); }

    /* status badge */
    .status-ok  { color: #4ADE80 !important; font-weight: 600; }
    .status-err { color: #F87171 !important; font-weight: 600; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1E3A5F, #2563EB);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 12px rgba(37,99,235,0.25);
    }
    [data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #BFDBFE !important; font-size: 0.8rem; }
    [data-testid="stMetricDelta"] { color: #93C5FD !important; font-size: 0.75rem; }

    /* ── Typography ── */
    h1 { color: #1E3A5F !important; font-weight: 700; letter-spacing: -0.5px; }
    h2 { color: #2563EB !important; font-weight: 600; }
    h3 { color: #1E40AF !important; }

    /* ── Dark background global ── */
    .stApp { background-color: #0F172A; }
    .main  { background-color: #0F172A; }

    /* ── Misc ── */
    .info-card {
        background: #1E293B; border: 1px solid #334155;
        border-radius: 12px; padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    .dataframe { font-size: 13px !important; }
    [data-testid="stFileUploader"] { border-radius: 10px; }

    /* ── Tab styling ── */
    [data-testid="stTabs"] button { color: #94A3B8 !important; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #60A5FA !important; border-bottom-color: #60A5FA !important; }
</style>
""", unsafe_allow_html=True)

# ── Google Drive Model IDs ─────────────────────────────────────────────────────
# Isi dengan File ID dari Google Drive setelah training selesai
GDRIVE_IDS = {
    'rf_model.pkl':        '',
    'xgb_model.pkl':       '',
    'scaler.pkl':          '',
    'feature_columns.pkl': '',
    'ann_model.h5':        '',
}

# ── Auto-download Model dari Google Drive ─────────────────────────────────────
def download_models_from_drive():
    try:
        import gdown
    except ImportError:
        return

    for fname, fid in GDRIVE_IDS.items():
        if not fid:
            continue
        if os.path.exists(os.path.join('models', fname)) or os.path.exists(fname):
            continue
        with st.spinner(f'Mengunduh {fname} dari Google Drive...'):
            try:
                gdown.download(f'https://drive.google.com/uc?id={fid}', fname, quiet=True)
            except Exception:
                pass

download_models_from_drive()

# ── Cari file model (root atau folder models/) ────────────────────────────────
def find_model_file(fname):
    if os.path.exists(fname):
        return fname
    alt = os.path.join('models', fname)
    if os.path.exists(alt):
        return alt
    return None

# ── Load Model ────────────────────────────────────────────────────────────────
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
    except Exception:
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

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_input(df: pd.DataFrame, models: dict) -> np.ndarray:
    if not models.get('prep_ok'):
        raise ValueError("Preprocessor tidak ditemukan. Jalankan notebook training terlebih dahulu.")

    df = df.copy()
    df.columns = df.columns.str.strip()
    df_num = df.select_dtypes(include=[np.number])

    feat_cols = models.get('feature_cols')
    if feat_cols:
        for col in feat_cols:
            if col not in df_num.columns:
                df_num[col] = 0.0
        df_num = df_num[feat_cols]

    df_num.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_num.fillna(0, inplace=True)

    return models['scaler'].transform(df_num.values)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### IDS Dashboard")
    st.markdown("**Machine Learning -- CIC-IDS-2017**")
    st.divider()
    halaman = st.radio(
        "Navigasi",
        ["Beranda", "Prediksi", "Evaluasi Model", "Eksplorasi Data"],
        label_visibility="collapsed"
    )
    st.divider()

    st.markdown("<p style='color:#94A3B8; font-size:0.75rem; margin-bottom:4px; text-transform:uppercase; letter-spacing:1px;'>STATUS MODEL</p>", unsafe_allow_html=True)
    for key, label in [('rf_ok', 'Random Forest'), ('ann_ok', 'ANN (Keras)'),
                        ('xgb_ok', 'XGBoost'), ('prep_ok', 'Preprocessor')]:
        ok     = models.get(key)
        cls    = "status-ok" if ok else "status-err"
        dot    = "&#9679;"
        status = "OK" if ok else "Tidak ditemukan"
        st.markdown(
            f'<div style="display:flex; justify-content:space-between; align-items:center; '
            f'padding:6px 10px; margin:3px 0; background:rgba(255,255,255,0.04); border-radius:8px;">'
            f'<span style="color:#CBD5E1; font-size:0.85rem;">{label}</span>'
            f'<span class="{cls}" style="font-size:0.8rem;">{dot} {status}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    if not any(models.get(k) for k in ['rf_ok', 'ann_ok', 'xgb_ok']):
        st.warning("Jalankan notebook training terlebih dahulu, lalu letakkan file .pkl & .h5 ke folder models/")

# ═══════════════════════════════════════════════════════════════════
# HALAMAN BERANDA
# ═══════════════════════════════════════════════════════════════════
if halaman == "Beranda":
    st.markdown("<h1 style='margin-bottom:0;'>Sistem Deteksi Intrusi Jaringan</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; margin-top:2px; margin-bottom:10px;'>Implementasi Machine Learning untuk Klasifikasi Serangan Siber &mdash; CIC-IDS-2017</p>", unsafe_allow_html=True)

    # Statistik ringkas
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Sampel",   "~2.8 Juta",  "CIC-IDS-2017")
    col2.metric("Data Training",  "~2 Juta",    "80% stratified")
    col3.metric("Data Testing",   "~504 Ribu",  "20% stratified")
    col4.metric("Jumlah Fitur",   "52",         "numerik")
    col5.metric("Kelas Serangan", "14 jenis",   "Binary: Normal vs Attack")

    st.divider()

    tab1, tab2 = st.tabs(["Tentang Sistem", "Kategori Serangan"])

    with tab1:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("""
            Dashboard ini merupakan implementasi **Intrusion Detection System (IDS)** berbasis
            Machine Learning menggunakan dataset **CIC-IDS-2017** dari Canadian Institute for
            Cybersecurity, University of New Brunswick.

            Sistem menggunakan tiga model utama yang dikombinasikan dalam **Ensemble Soft Voting**:
            - **Random Forest** — ensemble decision tree dengan class weighting
            - **XGBoost** — gradient boosting dengan scale_pos_weight
            - **ANN** — deep learning dengan BatchNormalization dan Dropout
            """)
        with c2:
            st.markdown("""
            **Fitur Utama**
            - Prediksi real-time via CSV
            - Evaluasi & perbandingan model
            - Eksplorasi data interaktif
            - Download hasil prediksi
            """)

    with tab2:
        attack_info = {
            "DoS / DDoS":       {"warna": "#EF4444", "deskripsi": "Membanjiri server hingga tidak dapat melayani pengguna sah.", "contoh": "DoS Hulk, GoldenEye, Slowloris, DDoS"},
            "Web Attack":       {"warna": "#F59E0B", "deskripsi": "Eksploitasi kerentanan aplikasi web untuk akses tidak sah.",  "contoh": "Brute Force, XSS, SQL Injection"},
            "Port Scan":        {"warna": "#8B5CF6", "deskripsi": "Pemindaian port untuk menemukan celah keamanan pada target.", "contoh": "PortScan"},
            "Infiltration/Bot": {"warna": "#EC4899", "deskripsi": "Penyusupan jaringan atau aktivitas botnet jarak jauh.",       "contoh": "Infiltration, Bot, Heartbleed"},
        }
        cols = st.columns(4)
        for i, (name, info) in enumerate(attack_info.items()):
            with cols[i]:
                st.markdown(f"""
                <div style="border-left:4px solid {info['warna']}; padding:10px 14px;
                            background:#1E293B; border-radius:8px; border:1px solid #334155;">
                    <b style="color:{info['warna']}; font-size:0.9rem;">{name}</b><br>
                    <small style="color:#CBD5E1;">{info['deskripsi']}</small><br>
                    <code style="font-size:10px; color:#94A3B8;">{info['contoh']}</code>
                </div>
                """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# HALAMAN PREDIKSI
# ═══════════════════════════════════════════════════════════════════
elif halaman == "Prediksi":
    st.title("Prediksi Intrusi Jaringan")
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
                              help="Probabilitas di atas nilai ini akan diklasifikasikan sebagai Attack")

    st.divider()

    feat_cols = models.get('feature_cols')
    with st.expander("Format File CSV yang Diperlukan"):
        if feat_cols:
            st.markdown(f"File CSV harus memiliki **{len(feat_cols)} kolom numerik** berikut (fitur CIC-IDS-2017):")
            st.code(", ".join(feat_cols))
        else:
            st.info("Load model terlebih dahulu untuk melihat daftar kolom yang diperlukan.")
        st.markdown("Kolom label (Attack Type, Label, dll.) tidak perlu disertakan.")

    uploaded_file = st.file_uploader("Upload File CSV", type=['csv', 'txt'])

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.success(f"File berhasil dimuat: **{df_input.shape[0]:,} baris, {df_input.shape[1]} kolom**")

            with st.expander("Preview Data (5 baris pertama)"):
                st.dataframe(df_input.head(), use_container_width=True)

            model_ready = (
                (model_choice == "Random Forest" and models.get('rf_ok')) or
                (model_choice == "ANN"           and models.get('ann_ok')) or
                (model_choice == "XGBoost"       and models.get('xgb_ok'))
            )

            if st.button("Jalankan Prediksi", type="primary", use_container_width=True):
                if not model_ready:
                    st.error(f"Model {model_choice} tidak tersedia. Letakkan file model ke folder models/.")
                else:
                    with st.spinner("Memproses data dan melakukan prediksi..."):
                        X = preprocess_input(df_input, models)

                        if model_choice == "Random Forest":
                            y_prob = models['rf'].predict_proba(X)[:, 1]
                        elif model_choice == "XGBoost":
                            y_prob = models['xgb'].predict_proba(X)[:, 1]
                        else:
                            y_prob = models['ann'].predict(X, verbose=0).flatten()

                        y_pred  = (y_prob >= threshold).astype(int)
                        labels  = ['Normal' if p == 0 else 'Attack' for p in y_pred]
                        n_normal = int(sum(y_pred == 0))
                        n_attack = int(sum(y_pred == 1))
                        total    = len(y_pred)

                        df_result = df_input.copy()
                        df_result['Prediksi']     = labels
                        df_result['Probabilitas'] = np.round(y_prob, 4)
                        df_result['Kelas']        = y_pred

                        st.divider()
                        st.markdown("### Hasil Prediksi")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Total Sampel", f"{total:,}")
                        m2.metric("Normal",       f"{n_normal:,}", f"{n_normal/total*100:.1f}%")
                        m3.metric("Attack",       f"{n_attack:,}", f"{n_attack/total*100:.1f}%")
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
                            "Download Hasil Prediksi (CSV)",
                            data=csv_out,
                            file_name='hasil_prediksi_ids.csv',
                            mime='text/csv',
                            type='primary'
                        )

        except Exception as e:
            st.error(f"Error memproses file: {e}")
            st.info("Pastikan file CSV berisi fitur numerik CIC-IDS-2017.")
    else:
        st.info("Upload file CSV untuk memulai prediksi.")

# ═══════════════════════════════════════════════════════════════════
# HALAMAN EVALUASI MODEL
# ═══════════════════════════════════════════════════════════════════
elif halaman == "Evaluasi Model":
    st.markdown("<h1 style='margin-bottom:0;'>Evaluasi dan Perbandingan Model</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; margin-top:2px; margin-bottom:10px;'>Perbandingan performa model machine learning pada dataset CIC-IDS-2017.</p>", unsafe_allow_html=True)

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
        return ['background-color: #166534; color: #FFFFFF; font-weight: bold' if v else 'color: #E2E8F0;' for v in is_max]

    tab1, tab2, tab3 = st.tabs(["Tabel & Grafik", "Confusion Matrix", "Feature Importance"])

    with tab1:
        col_tbl, col_chart = st.columns([1, 1])
        with col_tbl:
            styled = df_eval.style.apply(highlight_best).format("{:.4f}")
            st.dataframe(styled, use_container_width=True, height=245)
            st.caption("Terbaik: XGBoost (99.87%, ROC-AUC 1.0000)")
        with col_chart:
            fig, ax = plt.subplots(figsize=(6, 3.2))
            x = np.arange(len(df_eval.index))
            width = 0.18
            for i, (metric, color) in enumerate(zip(
                ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                ['#2563EB', '#16A34A', '#F59E0B', '#EF4444']
            )):
                ax.bar(x + i * width, df_eval[metric], width, label=metric, color=color, alpha=0.85)
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(df_eval.index, rotation=15, fontsize=7.5)
            ax.set_ylim(0.7, 1.05); ax.set_ylabel('Skor', fontsize=9)
            ax.set_title('Perbandingan Metrik', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8, ncol=2); ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with tab2:
        cm_file = st.file_uploader("Upload CSV data test (dengan kolom Label)", type=['csv'], key="cm_upload")
        if cm_file is not None and models.get('rf_ok') and models.get('prep_ok'):
            try:
                df_cm = pd.read_csv(cm_file)
                label_col = next((c for c in df_cm.columns if c.strip().lower() in ['label', 'attack type', ' label']), None)
                if label_col is None:
                    st.warning("Kolom label tidak ditemukan.")
                else:
                    from sklearn.metrics import confusion_matrix, classification_report
                    y_true = (df_cm[label_col].astype(str).str.strip().str.upper() != 'BENIGN').astype(int)
                    X_cm   = preprocess_input(df_cm.drop(columns=[label_col]), models)
                    y_pred = models['rf'].predict(X_cm)
                    cm_val = confusion_matrix(y_true, y_pred)
                    col_cm, col_cr = st.columns([1, 1])
                    with col_cm:
                        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                                    xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
                        ax_cm.set_xlabel('Prediksi', fontsize=9); ax_cm.set_ylabel('Aktual', fontsize=9)
                        ax_cm.set_title('Confusion Matrix -- RF', fontsize=10, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig_cm, use_container_width=True)
                        plt.close()
                    with col_cr:
                        st.markdown("**Classification Report**")
                        st.text(classification_report(y_true, y_pred, target_names=['Normal', 'Attack']))
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Upload CSV data test untuk melihat Confusion Matrix aktual.")

    with tab3:
        if models.get('rf_ok') and models.get('feature_cols'):
            feat_imp = pd.DataFrame({
                'Fitur':      models['feature_cols'],
                'Importance': models['rf'].feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            col_fi, _ = st.columns([2, 1])
            with col_fi:
                fig, ax = plt.subplots(figsize=(7, 3.5))
                colors = plt.cm.Blues(np.linspace(0.4, 0.9, 10))[::-1]
                bars = ax.barh(feat_imp['Fitur'][::-1], feat_imp['Importance'][::-1],
                               color=colors, edgecolor='#1E3A5F', linewidth=0.5)
                for bar, val in zip(bars, feat_imp['Importance'][::-1]):
                    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                            f'{val:.4f}', va='center', fontsize=8, fontweight='bold')
                ax.set_title('Top 10 Feature Importance -- Random Forest', fontsize=10, fontweight='bold')
                ax.set_xlabel('Importance Score', fontsize=9)
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
        else:
            st.info("Load model Random Forest untuk melihat Feature Importance aktual.")

# ═══════════════════════════════════════════════════════════════════
# HALAMAN EKSPLORASI DATA
# ═══════════════════════════════════════════════════════════════════
elif halaman == "Eksplorasi Data":
    st.markdown("<h1 style='margin-bottom:0;'>Eksplorasi Data CIC-IDS-2017</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; margin-top:2px; margin-bottom:10px;'>Visualisasi distribusi dan korelasi fitur dataset.</p>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Distribusi Label", "Deskripsi Fitur", "Heatmap Korelasi"])

    with tab1:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown("<small><b>Binary Label</b></small>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(3.5, 3))
            ax.pie([1446882, 1073708], labels=['Normal', 'Attack'],
                   autopct='%1.1f%%', colors=['#16A34A', '#DC2626'],
                   startangle=90, textprops={'fontsize': 9})
            ax.set_title('Distribusi Binary Label', fontsize=9, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with col2:
            st.markdown("<small><b>Per Jenis Serangan (Top 10)</b></small>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4, 3))
            cats   = ['BENIGN', 'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye',
                      'FTP-Patator', 'SSH-Patator', 'DoS Slowloris', 'DoS Slowhttptest', 'Bot']
            counts = [1446882, 231073, 158930, 128027, 10293, 7938, 5897, 5796, 5499, 1966]
            colors_bar = ['#16A34A'] + ['#EF4444'] * 9
            bars = ax.barh(cats[::-1], counts[::-1], color=colors_bar[::-1])
            for bar, val in zip(bars, counts[::-1]):
                ax.text(bar.get_width() + 1000, bar.get_y() + bar.get_height() / 2,
                        f'{val:,}', va='center', fontsize=7)
            ax.set_title('Top 10 Jenis Trafik', fontsize=9, fontweight='bold')
            ax.set_xlabel('Jumlah Sampel', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with col3:
            st.markdown("<small><b>Ringkasan Statistik</b></small>", unsafe_allow_html=True)
            st.markdown("""
            <div style='background:#1E293B; border-radius:8px; padding:12px; border:1px solid #334155;'>
            <table style='width:100%; font-size:0.78rem; color:#CBD5E1;'>
            <tr><td>Total sampel</td><td><b style='color:#60A5FA;'>2,830,743</b></td></tr>
            <tr><td>Normal (BENIGN)</td><td><b style='color:#4ADE80;'>1,446,882</b></td></tr>
            <tr><td>Attack</td><td><b style='color:#F87171;'>1,383,861</b></td></tr>
            <tr><td>Jenis serangan</td><td><b style='color:#FBBF24;'>14</b></td></tr>
            <tr><td>Jumlah fitur</td><td><b style='color:#A78BFA;'>52</b></td></tr>
            <tr><td>Sumber</td><td><b style='color:#94A3B8;'>CICFlowMeter</b></td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        col_feat, _ = st.columns([1, 1])
        with col_feat:
            if models.get('feature_cols'):
                feat_info = pd.DataFrame({
                    'No':    range(1, len(models['feature_cols']) + 1),
                    'Fitur': models['feature_cols'],
                    'Tipe':  ['Numerik'] * len(models['feature_cols']),
                })
                st.dataframe(feat_info, use_container_width=True, height=380)
                st.caption(f"Total {len(models['feature_cols'])} fitur numerik dari CICFlowMeter.")
            else:
                st.info("Load model untuk melihat daftar fitur aktual.")
                st.markdown("""
                **Jenis fitur CIC-IDS-2017:**
                - Statistik flow: durasi, panjang paket, jumlah paket
                - Flag TCP: SYN, ACK, FIN, RST, PSH, URG
                - Laju pengiriman: byte/s, packet/s
                - Window size, header length
                - IAT (Inter-Arrival Time): mean, std, max, min
                - Subflow: bytes dan packets forward/backward
                """)

    with tab3:
        st.caption("Ilustrasi korelasi — heatmap aktual tersedia setelah menjalankan notebook EDA.")
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
        col_hm, _ = st.columns([1, 1])
        with col_hm:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            mask = np.triu(np.ones_like(corr_df, dtype=bool))
            sns.heatmap(corr_df, mask=mask, annot=False, cmap='RdYlGn',
                        center=0, linewidths=0.2, ax=ax,
                        cbar_kws={'shrink': 0.5})
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
            ax.set_title('Heatmap Korelasi -- Top 15 Fitur (Ilustrasi)', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
