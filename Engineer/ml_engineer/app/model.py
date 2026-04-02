import pickle
import numpy as np
import face_recognition
import cv2
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

# ===================================
# Configuration
# ===================================
MODEL_PATH = Path(__file__).parent.parent / "models"
MODEL_FILE_PICKLE = MODEL_PATH / "face_knn_model.pkl"
MODEL_FILE_KERAS = MODEL_PATH / "face_cnn_model.keras"
THRESHOLD = 0.6  # Distance threshold for KNN

# -- module-level state
_knn = None
_cnn = None
_cnn_labels: list[str] = [] # Class index -> label name mapping

# ===================================
# Startup initialization
# ===================================

def load_models():
    global _knn, _cnn, _cnn_labels

    # Load KNN model
    if MODEL_FILE_PICKLE.exists():
        with open(MODEL_FILE_PICKLE, "rb") as f:
            data = pickle.load(f)
        _knn = data["knn"]
        print(f"[KNN] Model loaded - classes: {list(_knn.classes_)}")
    else:
        print(f"[KNN] Model file not found at {MODEL_FILE_PICKLE}")

    # Load CNN model
    if MODEL_FILE_KERAS.exists():
        _cnn = tf.keras.models.load_model(MODEL_FILE_KERAS)
        # Expect a labels saved alongside the model, e.g. in a .txt file with same name
        labels_file = MODEL_PATH / "face_cnn_model_info.txt"
        if labels_file.exists():
            _cnn_labels = labels_file.read_text().strip().splitlines()
            print(f"[CNN] Model loaded - classes: {_cnn_labels}")
    else:
        print(f"[CNN] Model file not found at {MODEL_FILE_KERAS}")

def is_ready() -> dict:
    return {
        "knn_loaded": _knn is not None,
        "cnn_loaded": _cnn is not None,
        "known_identities": list(_knn.classes_) if _knn else [],
        "threshold": THRESHOLD
    }

# ===================================
# Shared: ulities image bytes -> face encodings
# ===================================

def decode_image(image_bytes: bytes) -> np.ndarray:
    """Raw bytes -> RGB numpy array"""
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _detect_and_encode(rgb: np.ndarray):
    """
    Returns:
        encoding: np.ndarray (128-dim face embedding) or None
        bbox: [x, y, w, h] at original scale or None
        face_crop: 128x128 RGB crop for CNN or None"""
    small = cv2.resize(rgb, (0, 0), fx=0.25, fy=0.25)
    locs = face_recognition.face_locations(small, model="hog")
    encs = face_recognition.face_encodings(small, known_face_locations=locs)

    if not encs:
        return None, None, None
    
    # Scale bbbx back to original size
    top, right, bottom, left = locs[0]
    top, right, bottom, left = top*4, right*4, bottom*4, left*4
    bbox = [left, top, right-left, bottom-top]

    # Crop for CNN (padded, resized to 128x128)
    pad = 20
    h, w = rgb.shape[:2]
    y1 = max(0, top-pad); y2 = min(h, bottom+pad)
    x1 = max(0, left-pad); x2 = min(w, right+pad)
    face_crop = cv2.resize(rgb[y1:y2, x1:x2], (128, 128))

    return encs[0], bbox, face_crop

# ===================================
# KNN Prediction
# ===================================

def knn_predict(encoding: np.ndarray) -> dict:
    if _knn is None:
        return {"error": "KNN model not loaded"}
    
    dist, _ = _knn.kneighbors([encoding], n_neighbors=1)
    distance = float(dist[0][0])
    identity_knn = _knn.predict([encoding])[0]
    confidence = float(_knn.predict_proba([encoding]).max())

    if distance > THRESHOLD:
        identity_knn = "unknown"

    return {
        "identity": identity_knn,
        "confidence": round(confidence, 4),
        "distance": round(distance, 4),
        "model_used": "knn"
    }

# ===================================
# CNN Prediction
# ===================================

def cnn_predict(encoding: np.ndarray) -> dict:
    if _cnn is None:
        return {"error": "CNN model not loaded"}
    
    x = encoding.astype("float32")
    x = np.expand_dims(x, axis=0)

    prob = _cnn.predict(x, verbose=0)[0][0] # Binary classification: prob of "me"
    confidence = float(prob)
    identity = "me" if confidence >= 0.5 else "not_me"

    return {
        "identity": identity,
        "confidence": round(confidence, 4),
        "distance": round(1 - confidence, 4),
        "model_used": "cnn"
    }

# ===================================
# Ensemble: run both, weighted vote
# ===================================

def predict_ensemble(encoding: np.ndarray, face_crop: np.ndarray) -> dict:
    knn_result = knn_predict(encoding=encoding)
    cnn_result = cnn_predict(encoding=encoding)

    knn_ok = "error" not in knn_result
    cnn_ok = "error" not in cnn_result

    if not knn_ok and not cnn_ok:
        return {"error": "No models available"}
    
    # Weight: CNN gets 0.6, KNN gets 0.4 (if both available)
    scores: dict[str, float] = {}
    if knn_ok:
        label = knn_result["identity"]
        scores[label] = scores.get(label, 0) + knn_result["confidence"] * 0.4
    if cnn_ok:
        label = cnn_result["identity"]
        scores[label] = scores.get(label, 0) + cnn_result["confidence"] * 0.6

    best_identity = max(scores, key=scores.__getitem__)
    best_confidence = scores[best_identity]

    return {
        "identity": best_identity if best_confidence >= THRESHOLD else "unknown",
        "confidence": round(best_confidence, 4),
        "distance": knn_result.get("distance"),
        "model_used": "ensemble",
        "detail": {
            "knn": knn_result,
            "cnn": cnn_result
        }
    }

# ===================================
# Public API - Called by routes
# ===================================

def predict(image_bytes: bytes, mode: str = "ensemble") -> dict:
    """
    mode: "knn" | "cnn" | "ensemble"
    Returns a result dict or {"error": ...}
    """
    rgb = decode_image(image_bytes=image_bytes)
    encoding, bbox, face_crop = _detect_and_encode(rgb)

    if encoding is None:
        return {"error": "No face detected"}
    
    if mode == "knn":
        result = knn_predict(encoding=encoding)
    elif mode == "cnn":
        result = cnn_predict(encoding=encoding)
    else:
        result = predict_ensemble(encoding=encoding, face_crop=face_crop)

    result["bbox"] = bbox
    return result

def register(image_bytes: bytes, label: str) -> dict:
    """Add a new face encoding to KNN at runtime."""
    if _knn is None:
        return {"error": "KNN model not loaded"}
    
    rgb = decode_image(image_bytes=image_bytes)
    encoding, _, _ = _detect_and_encode(rgb)

    if encoding is None:
        return {"error": "No face detected"}
    
    X_new = np.vstack([_knn._fit_X, [encoding]])
    y_new = np.append(_knn._y, label)
    _knn.fit(X_new, y_new)

    return {"registered": label, "total_samples": len(X_new)}