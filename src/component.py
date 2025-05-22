import cv2
import numpy as np

def symmetry_score(roi):
    h, w = roi.shape[:2]
    left = roi[:, :w // 2]
    right = roi[:, w - w // 2:]
    right_flipped = cv2.flip(right, 1)
    diff = cv2.absdiff(left, right_flipped)
    score = np.mean(diff) / 255.0
    return score
