import cv2
import json
from src.detection import get_ped_bounding_box
from src.component import symmetry_score
from src.integrity import edge_density
from src.color_defect import spot_count
from src.classifier import load_model, classify_roi
from train_model import extract_features
def analyze_image(img_path, model_path='goz_pedi_svm.pkl'):
    # Görseli oku
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Görsel bulunamadı: {img_path}")

    # Ped kutusunu tespit et ve ROI çıkar
    bbox = get_ped_bounding_box(img)
    if bbox:
        x, y, w, h = bbox
        roi = img[y:y+h, x:x+w]
    else:
        roi = img  # fallback: tüm görüntü

    # Analizler
    sym_score = symmetry_score(roi)
    edge_dens = edge_density(roi)
    spot_num = spot_count(roi)
    model, classes = load_model(model_path)
    class_name, proba = classify_roi(roi, model, classes)

    # Sonucu döndür
    results = {
        "bounding_box": bbox,
        "symmetry_score": float(sym_score),
        "edge_density": float(edge_dens),
        "spot_count": int(spot_num),
        "predicted_class": class_name,
        "probabilities": [float(p) for p in proba]
    }
    return results

def classify_roi(roi, model, classes):
    feat = extract_features(roi).reshape(1, -1)  # Eğitimdekiyle birebir aynı!
    pred = model.predict(feat)[0]
    proba = model.predict_proba(feat)[0]
    return classes[pred], proba


if __name__ == "__main__":
    # Test amaçlı bir görsel dosyası
    img_path = "/Users/veysel/Desktop/hackathon/test/test.jpeg"
    output = analyze_image(img_path)
    print(json.dumps(output, indent=4, ensure_ascii=False))
