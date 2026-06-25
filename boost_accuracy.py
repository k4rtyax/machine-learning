# =============================================================================
# BOOST ACCURACY
# Jalankan cell-cell ini SETELAH bagian preprocessing (Bagian 5) di notebook
# Copy-paste per blok ke cell baru di notebook IDS_NSL_KDD.ipynb
# =============================================================================

# ===================== CELL 1: Install XGBoost =====================
# !pip install xgboost -q

# ===================== CELL 2: Import Tambahan =====================
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve
)
import xgboost as xgb

# ===================== CELL 3: Random Forest IMPROVED =====================
# KUNCI: class_weight='balanced' agar model tidak bias ke kelas mayoritas
# Ini mengatasi masalah RECALL RENDAH yang menyebabkan akurasi rendah

print('Training Random Forest (IMPROVED)...')
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# 5-Fold CV
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print(f'  CV Scores : {np.round(cv_scores, 4)}')
print(f'  Mean +- Std: {cv_scores.mean():.4f} +- {cv_scores.std():.4f}')

rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf  = f1_score(y_test, y_pred_rf)
print(f'RF Accuracy: {acc_rf:.4f} ({acc_rf*100:.2f}%)')
print(f'RF F1-Score: {f1_rf:.4f}')

joblib.dump(rf_model, 'rf_model.pkl')

# ===================== CELL 4: XGBoost =====================
print('\nTraining XGBoost...')

n_neg = sum(y_train == 0)
n_pos = sum(y_train == 1)
scale_ratio = n_neg / n_pos

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_ratio,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_prob_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

acc_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb  = f1_score(y_test, y_pred_xgb)
print(f'XGBoost Accuracy: {acc_xgb:.4f} ({acc_xgb*100:.2f}%)')
print(f'XGBoost F1-Score: {f1_xgb:.4f}')

# ===================== CELL 5: ANN IMPROVED =====================
print('\nTraining ANN (IMPROVED)...')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

n_features = X_train_scaled.shape[1]

from sklearn.utils.class_weight import compute_class_weight
class_weights_arr = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {0: class_weights_arr[0], 1: class_weights_arr[1]}
print(f'  Class weights: {class_weights}')

ann_model = Sequential([
    Dense(256, activation='relu', input_shape=(n_features,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])

ann_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

history = ann_model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=256,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

y_prob_ann = ann_model.predict(X_test_scaled, verbose=0).flatten()
y_pred_ann = (y_prob_ann >= 0.5).astype(int)

acc_ann = accuracy_score(y_test, y_pred_ann)
f1_ann  = f1_score(y_test, y_pred_ann)
print(f'ANN Accuracy: {acc_ann:.4f} ({acc_ann*100:.2f}%)')
print(f'ANN F1-Score: {f1_ann:.4f}')

ann_model.save('ann_model.h5')

# ===================== CELL 6: Model Pembanding =====================
print('\nTraining model pembanding...')

dt_model  = DecisionTreeClassifier(
    max_depth=25, min_samples_split=5, class_weight='balanced', random_state=42
)
nb_model  = GaussianNB()
knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

dt_model.fit(X_train_scaled, y_train)
nb_model.fit(X_train_scaled, y_train)
knn_model.fit(X_train_scaled, y_train)

y_pred_dt  = dt_model.predict(X_test_scaled)
y_pred_nb  = nb_model.predict(X_test_scaled)
y_pred_knn = knn_model.predict(X_test_scaled)

y_prob_dt  = dt_model.predict_proba(X_test_scaled)[:, 1]
y_prob_nb  = nb_model.predict_proba(X_test_scaled)[:, 1]
y_prob_knn = knn_model.predict_proba(X_test_scaled)[:, 1]

print('Semua model selesai.')

# ===================== CELL 7: Tabel Perbandingan =====================
def evaluate_model(y_true, y_pred, y_prob, model_name):
    return {
        'Model'    : model_name,
        'Accuracy' : round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall'   : round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1-Score' : round(f1_score(y_true, y_pred, zero_division=0), 4),
        'ROC-AUC'  : round(roc_auc_score(y_true, y_prob), 4)
    }

results = [
    evaluate_model(y_test, y_pred_rf,  y_prob_rf,  'Random Forest'),
    evaluate_model(y_test, y_pred_ann, y_prob_ann, 'ANN'),
    evaluate_model(y_test, y_pred_xgb, y_prob_xgb, 'XGBoost'),
    evaluate_model(y_test, y_pred_dt,  y_prob_dt,  'Decision Tree'),
    evaluate_model(y_test, y_pred_nb,  y_prob_nb,  'Naive Bayes'),
    evaluate_model(y_test, y_pred_knn, y_prob_knn, 'KNN'),
]

df_results = pd.DataFrame(results).set_index('Model')
print('=' * 70)
print('TABEL PERBANDINGAN PERFORMA MODEL (IMPROVED)')
print('=' * 70)
print(df_results.to_string())
print('=' * 70)

best = df_results['Accuracy'].idxmax()
print(f'\nMODEL TERBAIK: {best}')
print(f'   Accuracy : {df_results.loc[best, "Accuracy"]:.4f} ({df_results.loc[best, "Accuracy"]*100:.2f}%)')
print(f'   F1-Score : {df_results.loc[best, "F1-Score"]:.4f}')
print(f'   ROC-AUC  : {df_results.loc[best, "ROC-AUC"]:.4f}')
