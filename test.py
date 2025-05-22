import cv2
import joblib
import numpy as np

# Eğitimde kullandığın extract_features fonksiyonu:
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

# Modeli yükle
model_dict = joblib.load('goz_pedi_svm.pkl')
model = model_dict['model']
categories = model_dict['classes']

# Görseli yükle ve feature çıkar
img_path = '/Users/veysel/Desktop/hackathon/test/WhatsApp Image 2025-05-21 at 20.35.05.jpeg'
img = cv2.imread(img_path)
feat = extract_features(img).reshape(1, -1)

# Tahmin yap
pred = model.predict(feat)
print("Tahmin edilen sınıf:", categories[pred[0]])
