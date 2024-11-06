import mediapipe as mp
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import cv2
from reg import HandLandmarkProcessor

class HandGestureComparator:
    def __init__(self):
        self.processor = HandLandmarkProcessor()
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task'),
                num_hands=1))
            
    def load_and_detect(self, image_path: str) -> Tuple[mp.Image, mp.tasks.vision.HandLandmarkerResult, np.ndarray]:
        """載入圖片並進行手勢檢測"""
        image = mp.Image.create_from_file(image_path)
        detection_result = self.detector.detect(image)
        annotated_image = self.draw_landmarks_on_image(image.numpy_view(), detection_result)
        return image, detection_result, annotated_image
    
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        """在圖片上繪製關鍵點"""
        hand_landmarks_list = detection_result.hand_landmarks
        annotated_image = np.copy(rgb_image)
        
        if not hand_landmarks_list:
            return annotated_image
            
        # 繪製關鍵點和連接線
        for hand_landmarks in hand_landmarks_list:
            # 繪製關鍵點
            for landmark in hand_landmarks:
                landmark_px = np.multiply([landmark.x, landmark.y], 
                                        [annotated_image.shape[1], annotated_image.shape[0]]).astype(int)
                cv2.circle(annotated_image, tuple(landmark_px), 3, (0, 255, 0), -1)
                
            # 繪製連接線
            connections = mp.solutions.hands.HAND_CONNECTIONS
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_point = hand_landmarks[start_idx]
                end_point = hand_landmarks[end_idx]
                
                start_px = np.multiply([start_point.x, start_point.y], 
                                     [annotated_image.shape[1], annotated_image.shape[0]]).astype(int)
                end_px = np.multiply([end_point.x, end_point.y], 
                                   [annotated_image.shape[1], annotated_image.shape[0]]).astype(int)
                
                cv2.line(annotated_image, tuple(start_px), tuple(end_px), (0, 255, 0), 2)
        
        return annotated_image
    
    def compare_gestures(self, 
                        template_path: str, 
                        test_path: str, 
                        title1: str = "Template", 
                        title2: str = "Test Image",
                        show_plot: bool = True) -> Optional[float]:
        """比較兩個手勢圖片的相似度"""
        # 載入並檢測兩張圖片
        template_image, template_result, template_annotated = self.load_and_detect(template_path)
        test_image, test_result, test_annotated = self.load_and_detect(test_path)
        
        # 轉換檢測結果為數組
        template_landmarks = self.processor.convert_to_array(template_result)
        test_landmarks = self.processor.convert_to_array(test_result)
        
        # 檢查是否成功檢測到手勢
        if template_landmarks is None or test_landmarks is None:
            print("Failed to detect hand in one or both images")
            return None
            
        # 計算相似度
        similarity = F.cosine_similarity(
            torch.tensor(template_landmarks[0]), 
            torch.tensor(test_landmarks[0]), 
            dim=1
        ).mean().item()
        
        # 顯示比較結果
        if show_plot:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(template_annotated)
            ax[0].axis('off')
            ax[0].set_title(f'{title1}\nHandedness: {template_landmarks[1]}')
            
            ax[1].imshow(test_annotated)
            ax[1].axis('off')
            ax[1].set_title(f'{title2}\nHandedness: {test_landmarks[1]}')
            
            plt.suptitle(f'Similarity Score: {similarity:.4f}')
            plt.show()
            
        return similarity
    
    def batch_compare(self, template_dir: str, test_dir: str, number: int) -> dict:
        """批量比較特定數字的手勢"""
        template_path = f'{template_dir}/{number}.png'
        test_path = f'{test_dir}/{number}.png'
        
        similarity = self.compare_gestures(
            template_path,
            test_path,
            f'Template Number {number}',
            f'Test Number {number}'
        )
        
        return {
            'number': number,
            'similarity': similarity,
            'template_path': template_path,
            'test_path': test_path
        }

# 使用示例
def main():
    comparator = HandGestureComparator()
    
    # 單次比較
    similarity = comparator.compare_gestures(
        './chinese_number_gestures/3.png',
        './real_case/teacher_wu/wu_3.png',
        'Example Image 3',
        'Real Case Image 3'
    )
    print(f'Similarity score: {similarity:.4f}')
    
    # 批量比較
    result = comparator.batch_compare(
        './chinese_number_gestures',
        './real_case/teacher_wu',
        3
    )
    print(f"Batch comparison result: {result}")

if __name__ == "__main__":
    main()