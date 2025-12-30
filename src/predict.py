# src/predict.py
import cv2
import joblib
from dataset import split_plate_equally
from hog_feature import extract_hog

# 加载训练好的 SVM 模型
clf = joblib.load("models/svm_char.pkl")

def predict_plate(plate_img):
    """
    输入：BGR 车牌图像
    输出：预测的车牌字符串
    """
    chars = split_plate_equally(plate_img, plate_str="X"*7)  # 只用长度
    result = ""

    for char_img, _ in chars:
        feat = extract_hog(char_img).reshape(1, -1)
        pred = clf.predict(feat)[0]
        result += pred

    return result


if __name__ == "__main__":
    img = cv2.imread("test_plate.jpg")  # 换成你自己的测试图
    print("预测结果：", predict_plate(img))
