from flask import Flask, request, jsonify
import cv2
import numpy as np
import io
import time
from src.detection import get_ped_bounding_box
from src.component import symmetry_score
from src.integrity import edge_density
from src.color_defect import spot_count
from src.classifier import load_model

# Burayı ekle!
from ai.collect_results import add_result

app = Flask(__name__)

model, classes = load_model('../goz_pedi_svm.pkl')  # Kökten başlatıyorsan ../ gerek yok

def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (64*64)
    hist = cv2.calcHist([gray], [0], None, [32], [0,256]).flatten()
    hist = hist / hist.sum()
    flip = cv2.flip(gray, 1)
    symmetry = 1.0 - np.sum(np.abs(gray.astype(float) - flip.astype(float))) / (64*64*255)
    flat = gray.flatten() / 255.0
    features = np.concatenate([flat, hist, [edge_density, symmetry]])
    return features

def classify_roi(roi, model, classes):
    feat = extract_features(roi).reshape(1, -1)
    pred = model.predict(feat)[0]
    proba = model.predict_proba(feat)[0]
    return classes[pred], proba

@app.route("/analyze", methods=["POST"])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    bbox = get_ped_bounding_box(img)
    roi = img if bbox is None else img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    sym_score = symmetry_score(roi)
    edge_dens = edge_density(roi)
    spot_num = spot_count(roi)
    class_name, proba = classify_roi(roi, model, classes)
    results = {
        "bounding_box": bbox,
        "symmetry_score": float(sym_score),
        "edge_density": float(edge_dens),
        "spot_count": int(spot_num),
        "predicted_class": class_name,
        "probabilities": [float(p) for p in proba]
    }

    # ---------- HAVUZA EKLEMEK İÇİN ŞU SATIRI EKLE ----------
    add_result({
        "predicted_class": class_name,
        "symmetry_score": float(sym_score),
        "edge_density": float(edge_dens),
        "spot_count": int(spot_num),
        "timestamp": time.time()
    })
    # --------------------------------------------------------

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=500, debug=True)
