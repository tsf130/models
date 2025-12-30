# src/dataset.py
import cv2
import os

# ===== 自动定位项目根目录（关键）=====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANNOTATION_ROOT = os.path.join(PROJECT_ROOT, "data", "CBLPRD-330k_v1")
IMAGE_ROOT = os.path.join(ANNOTATION_ROOT, "CBLPRD-330k")



def load_annotations(txt_name):
    """
    txt_name: 'train.txt' / 'val.txt' / 'data.txt'
    """
    samples = []
    txt_path = os.path.join(ANNOTATION_ROOT, txt_name)

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            img_rel, plate_str, plate_type = line.strip().split()
            # img_rel 形如：CBLPRD-330k/000208356.jpg
            img_name = os.path.basename(img_rel)
            img_path = os.path.join(IMAGE_ROOT, img_name)
            samples.append((img_path, plate_str))
    return samples


def split_plate_equally(img, plate_str):
    """
    等宽切字符（CBLPRD-330k 稳定适用）
    """
    h, w = img.shape[:2]
    n = len(plate_str)
    char_w = w // n

    chars = []
    for i, ch in enumerate(plate_str):
        c = img[:, i * char_w:(i + 1) * char_w]
        c = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        c = cv2.resize(c, (32, 64))
        chars.append((c, ch))
    return chars
