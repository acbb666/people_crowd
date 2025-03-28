import os
import cv2
import torch
import numpy as np
import random  # 导入 random 模块
from flask import Flask, request, jsonify
from ultralytics import YOLO

# 初始化 Flask
app = Flask(__name__)

# 加载 YOLOv11 人脸检测模型
# 获取当前脚本所在文件夹目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov11l-face.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH).to(device)
model.eval()

def detect_faces(image_path):
    """检测图片中的人脸数量"""
    image = cv2.imread(image_path)
    if image is None:
        return -1  # 读取失败
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(image)  # 修复方法名
    face_count = len(results[0].boxes)  # 修复统计逻辑
    
    return face_count

@app.route("/detect", methods=["POST"])
def detect_api():
    """API 端点，处理前端上传的图片并检测人脸数量"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image_path = os.path.join("temp", file.filename)
    file.save(image_path)
    
    face_count = detect_faces(image_path)
    os.remove(image_path)  # 处理完后删除临时文件
    
    return jsonify({"face_count": face_count})

if __name__ == "__main__":
    # 本地测试
    test_folder = os.path.join(BASE_DIR, 'CrowdHuman', 'data', 'CrowdHuman_test', 'images_test')  # 修正路径
    if not os.path.exists(test_folder):
        print(f"Test folder {test_folder} does not exist.")
    else:
        all_images = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]
        selected_images = random.sample(all_images, 100)  # 随机抽取 100 张图片
        
        for image_name in selected_images:
            image_path = os.path.join(test_folder, image_name)
            print(f"Processing file: {image_path}")
            count = detect_faces(image_path)
            print(f"{image_name}: {count} faces detected")
    
    # 启动后端服务
    app.run(host="0.0.0.0", port=5000, debug=True)
