import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch

class GeometryAssessment:
    def __init__(self, model_path='./yolov8n.pt'):
        self.shapes = ['circle', 'square', 'triangle']
        
        if model_path and os.path.exists(model_path):
            self.yolo_model = YOLO(model_path)
        else:
            # 如果沒有預訓練模型，改用輪廓檢測
            self.yolo_model = None
            print("No YOLO model found, using contour detection instead.")
        
        # SIFT特徵檢測器
        self.sift = cv2.SIFT_create()
        
        # 載入或創建參考形狀範本
        self.templates = {
            'circle': self._create_circle_template(),
            'square': self._create_square_template(),
            'triangle': self._create_triangle_template()
        }

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
        """使用輪廓檢測來檢測和分類形狀"""
        # 轉換為灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 二值化
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # 找到輪廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_shapes = []
        
        for contour in contours:
            # 計算輪廓面積，過濾小的噪聲
            area = cv2.contourArea(contour)
            if area < 100:  # 可調整的閾值
                continue
                
            # 獲取邊界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 識別形狀類型
            shape_type = self._identify_shape(contour)
            
            # 計算置信度（基於形狀匹配度）
            confidence = self._calculate_shape_confidence(contour, shape_type)
            
            detected_shapes.append({
                'type': shape_type,
                'bbox': (x, y, x+w, y+h),
                'confidence': confidence,
                'contour': contour
            })
            
        return detected_shapes

    def _identify_shape(self, contour):
        """識別形狀類型"""
        # 近似多邊形
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        
        # 獲取頂點數量
        vertices = len(approx)
        
        # 計算圓形度
        area = cv2.contourArea(contour)
        if area > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
            
        # 根據特徵判斷形狀
        if 0.85 <= circularity <= 1.15:  # 圓形
            return 'circle'
        elif vertices == 4:  # 方形
            return 'square'
        elif vertices == 3:  # 三角形
            return 'triangle'
        else:
            # 如果無法確定，返回最接近的形狀
            return self._get_closest_shape(contour)

    def _calculate_shape_confidence(self, contour, shape_type):
        """計算形狀匹配的置信度"""
        # 根據形狀特徵計算置信度
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area == 0:
            return 0.0
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if shape_type == 'circle':
            # 圓形度越接近1，置信度越高
            confidence = 1 - abs(1 - circularity)
        elif shape_type == 'square':
            # 檢查四個角是否接近90度
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approx) == 4:
                confidence = self._calculate_square_confidence(approx)
            else:
                confidence = 0.0
        else:  # triangle
            # 檢查三個角的角度
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approx) == 3:
                confidence = self._calculate_triangle_confidence(approx)
            else:
                confidence = 0.0
                
        return min(max(confidence, 0.0), 1.0)

    def _calculate_square_confidence(self, approx):
        """計算方形的置信度"""
        # 檢查四個角是否接近90度
        angles = []
        for i in range(4):
            pt1 = approx[i][0]
            pt2 = approx[(i+1)%4][0]
            pt3 = approx[(i+2)%4][0]
            
            # 計算角度
            angle = self._calculate_angle(pt1, pt2, pt3)
            angles.append(angle)
            
        # 理想情況下所有角度應該是90度
        angle_confidence = 1 - sum([abs(angle - 90) for angle in angles]) / (4 * 90)
        return max(angle_confidence, 0.0)

    def _calculate_triangle_confidence(self, approx):
        """計算三角形的置信度"""
        # 計算三個角度
        angles = []
        for i in range(3):
            pt1 = approx[i][0]
            pt2 = approx[(i+1)%3][0]
            pt3 = approx[(i+2)%3][0]
            
            angle = self._calculate_angle(pt1, pt2, pt3)
            angles.append(angle)
            
        # 理想情況下三個角的和應該是180度
        angle_sum = sum(angles)
        confidence = 1 - abs(180 - angle_sum) / 180
        return max(confidence, 0.0)

    def _calculate_angle(self, pt1, pt2, pt3):
        """計算三點形成的角度"""
        v1 = pt1 - pt2
        v2 = pt3 - pt2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    def _get_closest_shape(self, contour):
        """獲取最接近的形狀類型"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area == 0:
            return 'circle'  # 預設值
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # 近似多邊形
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        vertices = len(approx)
        
        # 根據特徵選擇最接近的形狀
        if abs(circularity - 1) < 0.2:
            return 'circle'
        elif vertices >= 4:
            return 'square'
        else:
            return 'triangle'

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
        if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            
            # 應用Lowe's ratio測試
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            # 計算相似度分數
            similarity_score = len(good_matches) / max(len(matches), 1)
            
            # 添加形狀特定的評分邏輯
            if shape_type == 'circle':
                # 檢查圓形度
                contours, _ = cv2.findContours(shape_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        circularity_score = 1 - abs(1 - circularity)
                        similarity_score = (similarity_score + circularity_score) / 2
            
            elif shape_type == 'square':
                # 檢查角度和邊長比例
                contours, _ = cv2.findContours(shape_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(contour)
                    width, height = rect[1]
                    if min(width, height) > 0:
                        aspect_ratio = max(width, height) / min(width, height)
                        aspect_score = 1 - min(abs(aspect_ratio - 1), 1)
                        similarity_score = (similarity_score + aspect_score) / 2
            
            elif shape_type == 'triangle':
                # 檢查三角形的規則性
                contours, _ = cv2.findContours(shape_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                    if len(approx) == 3:
                        # 計算三個邊的長度
                        sides = []
                        for i in range(3):
                            p1 = approx[i][0]
                            p2 = approx[(i + 1) % 3][0]
                            side = np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
                            sides.append(side)
                        # 計算邊長比例的一致性
                        max_side = max(sides)
                        if max_side > 0:
                            side_ratios = [side/max_side for side in sides]
                            ratio_score = 1 - max(abs(ratio - 1) for ratio in side_ratios)
                            similarity_score = (similarity_score + ratio_score) / 2
            
            return min(max(similarity_score, 0.0), 1.0)
        
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
        # 調整權重
        detection_weight = 0.4
        similarity_weight = 0.6
        
        # 基礎分數 (60-100分範圍)
        base_score = 60
        
        # 計算檢測分數 (0-20分)
        detection_score = detection_conf * 20
        
        # 計算相似度分數 (0-20分)
        similarity_score = similarity * 20
        
        # 計算總分
        final_score = base_score + (detection_score * detection_weight + similarity_score * similarity_weight)
        
        # 確保分數在0-100之間
        final_score = np.clip(final_score, 0, 100)
        
        return round(final_score, 2)

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