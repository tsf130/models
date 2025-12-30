# src/ui.py
import gradio as gr
import cv2
import joblib
import numpy as np
from dataset import split_plate_equally
from hog_feature import extract_hog

# 加载模型
clf = joblib.load("models/svm_char.pkl")

def visualize_chars(chars):
    """
    chars: [(char_img, label), ...]
    返回：一张横向拼接的 RGB 图像，便于 UI 显示
    """
    vis_list = []

    for char_img, _ in chars:
        # char_img 是灰度图，转为 3 通道
        rgb = cv2.cvtColor(char_img, cv2.COLOR_GRAY2RGB)
        vis_list.append(rgb)

    # 横向拼接
    vis_img = cv2.hconcat(vis_list)
    return vis_img

def recognize_plate(image, char_len):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    chars = split_plate_equally(image, plate_str="X" * char_len)

    plate_result = ""
    detail_result = []

    for idx, (char_img, _) in enumerate(chars):
        feat = extract_hog(char_img).reshape(1, -1)
        pred = clf.predict(feat)[0]

        score = clf.decision_function(feat)
        conf = float(np.max(score))

        plate_result += pred
        detail_result.append(f"第{idx+1}位：{pred}（置信度 {conf:.2f}）")

    detail_text = "\n".join(detail_result)

    # ★ 新增：字符切分可视化
    vis_img = visualize_chars(chars)

    return plate_result, detail_text, vis_img

demo = gr.Interface(
    fn=recognize_plate,
    inputs=[
        gr.Image(type="numpy", label="上传车牌图片"),
        gr.Radio(
                choices=[7, 8],
                value=7,
                label="车牌字符数（普通车牌选 7，新能 源车牌选 8）"
        )
    ],
    outputs=[
        gr.Textbox(label="识别结果"),
        gr.Textbox(label="逐字符识别结果（含置信度）", lines=8),
        gr.Image(type="numpy", label="字符切分可视化")
    ],
    title="基于机器学习的车牌识别系统（HOG + SVM）",
    description=(
        "功能说明：\n"
        "1. 系统对车牌图像进行等宽字符分割；\n"
        "2. 对每个字符提取 HOG 特征；\n"
        "3. 使用 SVM 分类器进行字符识别。\n\n"
        "注：置信度为字符样本到分类超平面的距离，用于反映预测可信程度。"
    ),
    submit_btn="开始识别",
    clear_btn="清空"
)

if __name__ == "__main__":
    demo.launch()
