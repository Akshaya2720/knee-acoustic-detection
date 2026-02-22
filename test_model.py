"""Quick end-to-end test of the model pipeline."""
import zipfile, tempfile, os, sys
sys.path.insert(0, r'c:\Users\user\Downloads\DUK\web_app\backend')

import joblib, numpy as np, librosa, warnings
warnings.filterwarnings('ignore')

from feature_extractor import extract_features, SR, MIN_SAMPLES

base = r'c:\Users\user\Downloads\DUK\web_app\backend'
ZIP  = r'C:\Users\user\Downloads\converted_wav_data.zip'

print("=== Loading model ===")
model  = joblib.load(os.path.join(base, 'knee_model.pkl'))
scaler = joblib.load(os.path.join(base, 'knee_scaler.pkl'))
print(f"Model   : {type(model).__name__}")
print(f"Features: {scaler.n_features_in_}")

def predict_wav(path):
    sig, _ = librosa.load(path, sr=SR, mono=True)
    print(f"  Signal length: {len(sig)} samples ({len(sig)/SR*1000:.1f} ms)")
    if len(sig) < MIN_SAMPLES:
        repeats = int(np.ceil(MIN_SAMPLES / len(sig)))
        sig = np.tile(sig, repeats)[:MIN_SAMPLES]
        print(f"  Tiled to: {len(sig)} samples")
    feats = extract_features(sig)
    print(f"  Feature shape: {feats.shape}")
    feats_sc = scaler.transform(feats.reshape(1, -1))
    pred = int(model.predict(feats_sc)[0])
    prob = model.predict_proba(feats_sc)[0]
    label = 'Normal' if pred == 0 else 'Abnormal'
    return pred, prob, label

print("\n=== Testing with real WAV files ===")
with zipfile.ZipFile(ZIP) as z:
    names = z.namelist()
    normal_wavs   = [n for n in names if n.startswith('0/') and n.endswith('.wav')][:3]
    abnormal_wavs = [n for n in names if n.startswith('1/') and n.endswith('.wav')][:3]

    print(f"\n--- Normal samples (expected: Normal / class 0) ---")
    for name in normal_wavs:
        with z.open(name) as f:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(f.read()); tmp_path = tmp.name
        pred, prob, label = predict_wav(tmp_path)
        os.unlink(tmp_path)
        ok = '✅' if pred == 0 else '❌ WRONG'
        print(f"  {ok}  {name:<20} → {label}  (conf={prob[pred]*100:.1f}%)")

    print(f"\n--- Abnormal samples (expected: Abnormal / class 1) ---")
    for name in abnormal_wavs:
        with z.open(name) as f:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(f.read()); tmp_path = tmp.name
        pred, prob, label = predict_wav(tmp_path)
        os.unlink(tmp_path)
        ok = '✅' if pred == 1 else '❌ WRONG'
        print(f"  {ok}  {name:<20} → {label}  (conf={prob[pred]*100:.1f}%)")

print("\n=== Done ===")
