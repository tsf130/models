# src/svm_train.py
import os
import cv2
import joblib
import numpy as np
from sklearn.svm import SVC
from dataset import load_annotations, split_plate_equally
from hog_feature import extract_hog


from dataset import load_annotations

samples = load_annotations("train.txt")


X, y = [], []


print("加载训练样本中...")
for img_path, plate_str in samples[:30000]:  # 可先用 3 万张跑通
    img = cv2.imread(img_path)
    if img is None:
        continue

    chars = split_plate_equally(img, plate_str)
    for char_img, label in chars:
        feat = extract_hog(char_img)
        X.append(feat)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("开始训练 SVM...")
clf = SVC(kernel="linear", C=1.0)
clf.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/svm_char.pkl")
print("模型已保存：models/svm_char.pkl")
