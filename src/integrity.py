import cv2
import numpy as np

def edge_density(roi):
    edges = cv2.Canny(roi, 50, 150)
    density = np.sum(edges > 0) / edges.size
    return density
