"""
train_model.py
──────────────────────────────────────────────────────
Trains the KneeGuard AI classifier on WAV files from
converted_wav_data.zip and saves model artifacts to
web_app/backend/.

Run: python train_model.py
──────────────────────────────────────────────────────
"""
import os, sys, json, zipfile, tempfile, time, warnings
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────
ZIP_PATH    = r"C:\Users\user\Downloads\converted_wav_data.zip"
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "web_app", "backend")
SR          = 22050
MIN_SAMPLES = 4096

print("=" * 58)
print("  KneeGuard AI — Local Training Script")
print("=" * 58)

# ─── Feature Extraction (identical to feature_extractor.py) ──
def preprocess_signal(signal):
    signal = signal.astype(np.float32)
    if len(signal) < MIN_SAMPLES:
        repeats = int(np.ceil(MIN_SAMPLES / len(signal)))
        signal = np.tile(signal, repeats)[:MIN_SAMPLES]
    return signal

def extract_features(signal, sr=SR):
    signal = preprocess_signal(signal)
    if signal.max() != signal.min():
        signal = (signal - signal.mean()) / (signal.std() + 1e-9)
    features = []
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1)); features.extend(np.std(mfcc, axis=1))
    sc = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    features.extend([np.mean(sc), np.std(sc)])
    sb = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
    features.extend([np.mean(sb), np.std(sb)])
    sr_feat = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    features.extend([np.mean(sr_feat), np.std(sr_feat)])
    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    features.extend([np.mean(zcr), np.std(zcr)])
    rms = librosa.feature.rms(y=signal)[0]
    features.extend([np.mean(rms), np.std(rms)])
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    features.extend(np.mean(chroma, axis=1)); features.extend(np.std(chroma, axis=1))
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
    features.extend(np.mean(contrast, axis=1)); features.extend(np.std(contrast, axis=1))
    return np.array(features, dtype=np.float32)

# ─── Load WAV files from zip ──────────────────────────
print(f"\n📦 Loading WAV files from zip...")
X, y = [], []
errors = 0

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    all_names = z.namelist()
    wav_files = [(n, int(n.split('/')[0])) for n in all_names
                 if n.endswith('.wav') and n.split('/')[0] in ('0', '1')]

    total = len(wav_files)
    print(f"   Found {total} WAV files  "
          f"(Normal={sum(1 for _,l in wav_files if l==0)}, "
          f"Abnormal={sum(1 for _,l in wav_files if l==1)})")

    for i, (name, label) in enumerate(wav_files, 1):
        if i % 200 == 0 or i == total:
            pct = i / total * 100
            bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            print(f"\r   [{bar}] {pct:5.1f}%  ({i}/{total})", end='', flush=True)
        try:
            with z.open(name) as f:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
            sig, _ = librosa.load(tmp_path, sr=SR, mono=True)
            os.unlink(tmp_path)
            if len(sig) == 0:
                continue
            X.append(extract_features(sig))
            y.append(label)
        except Exception as e:
            errors += 1

print(f"\n   ✅ Loaded {len(X)} samples  ({errors} errors skipped)")

X = np.array(X)
y = np.array(y)
print(f"   Dataset shape: X={X.shape}, y={y.shape}")
print(f"   Class dist   : Normal={sum(y==0)}, Abnormal={sum(y==1)}")

# ─── Split & Scale ───────────────────────────────────
print("\n📊 Splitting and scaling...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# SMOTE
k = min(5, int(sum(y_train == 1)) - 1)
smote = SMOTE(random_state=42, k_neighbors=max(1, k))
X_bal, y_bal = smote.fit_resample(X_train_sc, y_train)
print(f"   After SMOTE : Normal={sum(y_bal==0)}, Abnormal={sum(y_bal==1)}")

# ─── Train Models ────────────────────────────────────
print("\n🤖 Training models...\n")

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
    ),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"⏳ {name}...")
    t0 = time.time()
    scores = cross_val_score(model, X_bal, y_bal, cv=cv, scoring="f1_weighted", n_jobs=-1)
    model.fit(X_bal, y_bal)
    elapsed = time.time() - t0

    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    report = classification_report(y_test, y_pred,
                                   target_names=["Normal", "Abnormal"], output_dict=True)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = dict(
        model=model,
        cv_f1=scores.mean(), cv_std=scores.std(),
        test_acc=report["accuracy"],
        test_f1=report["weighted avg"]["f1-score"],
        abnormal_recall=report["Abnormal"]["recall"],
        auc=auc, time=elapsed
    )
    print(f"   Accuracy={report['accuracy']*100:.1f}%  "
          f"AUC={auc:.3f}  Abnormal-Recall={report['Abnormal']['recall']*100:.1f}%  "
          f"({elapsed:.0f}s)\n")

# ─── Select Best ─────────────────────────────────────
best_name = max(results, key=lambda n: (results[n]["auc"], results[n]["abnormal_recall"]))
best_model = results[best_name]["model"]

print("=" * 58)
print(f"  🏆 BEST MODEL   : {best_name}")
print(f"  Accuracy        : {results[best_name]['test_acc']*100:.2f}%")
print(f"  ROC-AUC         : {results[best_name]['auc']:.4f}")
print(f"  Abnormal Recall : {results[best_name]['abnormal_recall']*100:.1f}%")
print("=" * 58)

y_pred_best = best_model.predict(X_test_sc)
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=["Normal", "Abnormal"]))

# ─── Save Artifacts ──────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_path  = os.path.join(OUTPUT_DIR, "knee_model.pkl")
scaler_path = os.path.join(OUTPUT_DIR, "knee_scaler.pkl")
meta_path   = os.path.join(OUTPUT_DIR, "knee_model_meta.json")

joblib.dump(best_model, model_path)
joblib.dump(scaler,     scaler_path)

meta = {
    "model_name"      : best_name,
    "feature_count"   : int(X.shape[1]),
    "sample_rate"     : SR,
    "min_samples"     : MIN_SAMPLES,
    "classes"         : {"0": "Normal", "1": "Abnormal"},
    "test_accuracy"   : float(results[best_name]["test_acc"]),
    "roc_auc"         : float(results[best_name]["auc"]),
    "abnormal_recall" : float(results[best_name]["abnormal_recall"]),
    "trained_on"      : "wav_files",
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✅ Model saved to: {OUTPUT_DIR}")
print(f"   knee_model.pkl       ({os.path.getsize(model_path)//1024} KB)")
print(f"   knee_scaler.pkl      ({os.path.getsize(scaler_path)//1024} KB)")
print(f"   knee_model_meta.json")
print(f"\n🎉 Done! Restart uvicorn and test the web app.")
print(f"   cd web_app\\backend")
print(f"   uvicorn app:app --reload --port 8000")
