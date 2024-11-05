import cv2
import numpy as np
from ultralytics import YOLO
import torch

class GeometryAssessment:
    def __init__(self):
        # 初始化YOLO模型
        self.yolo_model = YOLO('yolov8n.pt')  # 使用預訓練模型作為基礎
        self.shapes = ['circle', 'square', 'triangle']
        
        # 載入或創建參考形狀範本
        self.templates = {
            'circle': self._create_circle_template(),
            'square': self._create_square_template(),
            'triangle': self._create_triangle_template()
        }
        
        # SIFT特徵檢測器
        self.sift = cv2.SIFT_create()

    def _create_circle_template(self):
        """創建標準圓形範本"""
        template = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(template, (100, 100), 80, 255, 2)
        return template

    def _create_square_template(self):
        """創建標準方形範本"""
        template = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(template, (40, 40), (160, 160), 255, 2)
        return template

    def _create_triangle_template(self):
        """創建標準三角形範本"""
        template = np.zeros((200, 200), dtype=np.uint8)
        pts = np.array([[100, 40], [40, 160], [160, 160]], np.int32)
        cv2.polylines(template, [pts], True, 255, 2)
        return template

    def preprocess_image(self, image):
        """預處理學生繪製的圖片"""
        # 轉換為灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 二值化處理
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # 去噪
        binary = cv2.GaussianBlur(binary, (5, 5), 0)
        return binary

    def detect_shapes(self, image):
        """使用YOLO檢測圖形"""
        results = self.yolo_model(image)
        detected_shapes = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_id = box.cls[0]
                shape_type = self.shapes[int(class_id)]
                detected_shapes.append({
                    'type': shape_type,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(confidence)
                })
        return detected_shapes

    def calculate_shape_similarity(self, image, shape_type, bbox):
        """使用SIFT計算形狀相似度"""
        # 裁剪檢測到的形狀
        x1, y1, x2, y2 = bbox
        shape_img = image[y1:y2, x1:x2]
        shape_img = cv2.resize(shape_img, (200, 200))
        shape_img = self.preprocess_image(shape_img)
        
        # 獲取模板
        template = self.templates[shape_type]
        
        # 計算SIFT特徵
        kp1, des1 = self.sift.detectAndCompute(shape_img, None)
        kp2, des2 = self.sift.detectAndCompute(template, None)
        
        # 特徵匹配
        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            
            # 應用Lowe's ratio測試
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            similarity_score = len(good_matches) / len(matches)
            return similarity_score
        return 0.0

    def assess_drawing(self, image):
        """評估學生繪畫"""
        # 檢測形狀
        detected_shapes = self.detect_shapes(image)
        assessment_results = []
        
        for shape in detected_shapes:
            # 計算形狀相似度
            similarity = self.calculate_shape_similarity(
                image, 
                shape['type'], 
                shape['bbox']
            )
            
            # 評分標準
            score = self._calculate_score(
                shape['confidence'],
                similarity
            )
            
            assessment_results.append({
                'shape_type': shape['type'],
                'detection_confidence': shape['confidence'],
                'shape_similarity': similarity,
                'score': score,
                'feedback': self._generate_feedback(score)
            })
            
        return assessment_results

    def _calculate_score(self, detection_conf, similarity):
        """計算最終分數"""
        # 綜合YOLO的檢測信心度和SIFT的相似度
        weighted_score = (detection_conf * 0.4 + similarity * 0.6) * 100
        return min(round(weighted_score, 2), 100)

    def _generate_feedback(self, score):
        """根據分數生成回饋建議"""
        if score >= 90:
            return "優秀！形狀畫得非常準確"
        elif score >= 75:
            return "不錯！形狀基本正確，但還可以更準確"
        elif score >= 60:
            return "及格。建議多練習，提高準確度"
        else:
            return "需要加強。建議參考標準範本多加練習"