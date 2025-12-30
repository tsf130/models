# src/svm_eval.py
import cv2
import joblib
from dataset import load_annotations, split_plate_equally
from hog_feature import extract_hog

def main():
    # 1. 加载验证集标注
    samples = load_annotations("val.txt")

    # 2. 加载训练好的 SVM 模型
    clf = joblib.load("models/svm_char.pkl")

    # 3. 初始化总体统计
    total, correct = 0, 0

    # 4. 初始化分字符类别统计
    # 每一项格式：[correct, total]
    stats = {
        "digit": [0, 0],     # 数字
        "letter": [0, 0],    # 英文字母
        "chinese": [0, 0]    # 汉字（省份简称等）
    }

    # 5. 遍历验证集样本（可限制数量以加快评估）
    for img_path, plate_str in samples[:5000]:
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 6. 字符分割（与训练阶段保持一致）
        chars = split_plate_equally(img, plate_str)

        for char_img, gt in chars:
            # 7. 特征提取
            feat = extract_hog(char_img).reshape(1, -1)

            # 8. 字符预测
            pred = clf.predict(feat)[0]

            # ===== 总体统计 =====
            total += 1
            if pred == gt:
                correct += 1

            # ===== 分类别统计 =====
            if gt.isdigit():
                key = "digit"
            elif gt.isalpha():
                key = "letter"
            else:
                key = "chinese"

            stats[key][1] += 1
            if pred == gt:
                stats[key][0] += 1

    # 9. 输出评估结果
    if total == 0:
        print("没有有效的验证样本")
        return

    print(f"总体字符级准确率：{correct / total:.4f}")

    print("各字符类别识别准确率：")
    for k, (c, t) in stats.items():
        if t > 0:
            print(f"  {k}: {c / t:.4f} ({c}/{t})")
        else:
            print(f"  {k}: 无样本")

if __name__ == "__main__":
    main()
