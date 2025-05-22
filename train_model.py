import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# 1. Dosya yolu ve sınıflar
base_root = r"/Users/veysel/Desktop/hackathon/dataset"
categories = [
    "clean",
    "integrity_check",
    "color_defect_detection"
]

def extract_features(img):
    # 1) Gri seviyeye çevir ve normalize et
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))

    # 2) Kenar yoğunluğu (Canny kenar tespiti)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (64*64)

    # 3) Histogram (gri seviye)
    hist = cv2.calcHist([gray], [0], None, [32], [0,256]).flatten()
    hist = hist / hist.sum()  # normalize

    # 4) Görüntünün simetrisi (ayna ile fark)
    flip = cv2.flip(gray, 1)
    symmetry = 1.0 - np.sum(np.abs(gray.astype(float) - flip.astype(float))) / (64*64*255)

    # 5) Flat vektör
    flat = gray.flatten() / 255.0

    # 6) Hepsini birleştir
    features = np.concatenate([flat, hist, [edge_density, symmetry]])
    return features

# 2. Görüntüleri oku ve öznitelikleri çıkar
X, y = [], []
for idx, cls in enumerate(categories):
    folder = os.path.join(base_root, cls)
    for fname in os.listdir(folder):
        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        feat = extract_features(img)
        X.append(feat)
        y.append(idx)

X = np.array(X)
y = np.array(y)
print(f"Toplam örnek: {len(y)}, Sınıf dağılımı: {np.bincount(y)}")

# 3. Eğitim/test böl
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Eğitim: {len(y_train)}, Test: {len(y_test)}")

# 4. SVM eğitimi
model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X_train, y_train)
print("Model eğitildi.")

# 5. Değerlendirme
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=categories))

# 6. Modeli kaydet
joblib.dump({'model': model, 'classes': categories}, 'goz_pedi_svm.pkl')
print("Model kaydedildi: goz_pedi_svm.pkl")
