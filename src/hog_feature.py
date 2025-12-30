# src/hog_feature.py
import cv2

def extract_hog(img):
    hog = cv2.HOGDescriptor(
        _winSize=(32, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    feature = hog.compute(img)
    return feature.flatten()
