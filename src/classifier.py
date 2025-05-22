import cv2
import numpy as np
import joblib

def load_model(pkl_path):
    data = joblib.load(pkl_path)
    return data['model'], data['classes']

def classify_roi(roi, model, classes):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    feat = cv2.resize(gray, (64, 64)).flatten().reshape(1, -1)
    pred = model.predict(feat)[0]
    proba = model.predict_proba(feat)[0]
    return classes[pred], proba
