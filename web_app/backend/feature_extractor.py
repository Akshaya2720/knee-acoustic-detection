"""
feature_extractor.py
──────────────────────────────────────────────────────────────────
⚠️  IMPORTANT: This function is an EXACT MIRROR of the one in
    knee_classifier_colab.ipynb.  Any changes here must be
    reflected in the notebook and vice-versa.
──────────────────────────────────────────────────────────────────
"""
import numpy as np
import librosa

SR = 22050          # Fixed sample rate used during training
MIN_SAMPLES = 4096  # Minimum signal length — SHORT signals are TILED (not zero-padded)
                    # Same constant used in app.py and Colab notebook


def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    """
    Ensure signal is at least MIN_SAMPLES long by TILING (repeating).

    Zero-padding was intentionally avoided: padding with silence distorts
    spectral features (centroid, rolloff, chroma) — causing every short clip
    to look identical to the model.  Tiling preserves the spectral content of
    the original short knee acoustic burst.
    """
    signal = signal.astype(np.float32)
    if len(signal) < MIN_SAMPLES:
        repeats = int(np.ceil(MIN_SAMPLES / len(signal)))
        signal = np.tile(signal, repeats)[:MIN_SAMPLES]
    return signal


def extract_features(signal: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Extract a fixed-length feature vector from a 1-D acoustic signal.

    Parameters
    ----------
    signal : np.ndarray  — raw audio samples (float32/float64)
    sr     : int         — sample rate (default matches training SR)

    Returns
    -------
    np.ndarray of shape (72,)  — dtype float32
    """
    # 1. Ensure minimum length via tiling
    signal = preprocess_signal(signal)

    # 2. Z-score normalize
    if signal.max() != signal.min():
        signal = (signal - signal.mean()) / (signal.std() + 1e-9)

    features = []

    # 3. MFCC (13 × mean+std = 26)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # 4. Spectral Centroid (2)
    sc = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    features.extend([np.mean(sc), np.std(sc)])

    # 5. Spectral Bandwidth (2)
    sb = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
    features.extend([np.mean(sb), np.std(sb)])

    # 6. Spectral Rolloff (2)
    sr_feat = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    features.extend([np.mean(sr_feat), np.std(sr_feat)])

    # 7. Zero Crossing Rate (2)
    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    features.extend([np.mean(zcr), np.std(zcr)])

    # 8. RMS Energy (2)
    rms = librosa.feature.rms(y=signal)[0]
    features.extend([np.mean(rms), np.std(rms)])

    # 9. Chroma STFT (12 × mean+std = 24)
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # 10. Spectral Contrast (7 × mean+std = 14)
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    return np.array(features, dtype=np.float32)
