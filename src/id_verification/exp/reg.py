import numpy as np
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from typing import Optional, Tuple

class HandLandmarkProcessor:
    def __init__(self):
        # 手部關鍵點的數量
        self.NUM_LANDMARKS = 21
        
    def convert_to_array(self, result: HandLandmarkerResult) -> Optional[Tuple[np.ndarray, str]]:
        """
        將 HandLandmarkerResult 轉換為 numpy array
        
        Args:
            result: MediaPipe HandLandmarkerResult 物件
            
        Returns:
            Tuple[np.ndarray, str]: (landmarks_array, handedness)
            - landmarks_array: shape (21, 3) 的 numpy array，包含 x,y,z 座標
            - handedness: 'Left' 或 'Right'
            如果沒有檢測到手，返回 None
        """
        # 檢查是否有檢測到手
        if not result.handedness or not result.hand_landmarks:
            return None
            
        # 獲取第一隻手的資訊（假設只檢測一隻手）
        handedness = result.handedness[0][0].category_name  # 'Left' 或 'Right'
        landmarks = result.hand_landmarks[0]  # 第一隻手的關鍵點
        
        # 轉換為 numpy array
        landmarks_array = np.zeros((self.NUM_LANDMARKS, 3))
        for i, landmark in enumerate(landmarks):
            landmarks_array[i] = [landmark.x, landmark.y, landmark.z]
            
        return landmarks_array, handedness
        
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        正規化關鍵點座標
        
        Args:
            landmarks: shape (21, 3) 的 numpy array
            
        Returns:
            正規化後的 numpy array
        """
        # 1. 中心化 - 使用手腕(索引0)作為參考點
        centered = landmarks - landmarks[0]
        
        # 2. 縮放正規化
        scale = np.max(np.linalg.norm(centered, axis=1))
        normalized = centered / scale
        
        return normalized
        
    def extract_features(self, result: HandLandmarkerResult) -> Optional[np.ndarray]:
        """
        從 HandLandmarkerResult 提取特徵向量
        
        Args:
            result: MediaPipe HandLandmarkerResult 物件
            
        Returns:
            特徵向量 numpy array，如果沒有檢測到手則返回 None
        """
        conversion_result = self.convert_to_array(result)
        if conversion_result is None:
            return None
            
        landmarks_array, handedness = conversion_result
        
        # 正規化關鍵點
        normalized_landmarks = self.normalize_landmarks(landmarks_array)
        
        # 計算指尖之間的距離
        fingertips = [4, 8, 12, 16, 20]  # 拇指、食指、中指、無名指、小指的指尖索引
        distances = []
        for i in range(len(fingertips)):
            for j in range(i + 1, len(fingertips)):
                dist = np.linalg.norm(
                    normalized_landmarks[fingertips[i]] - normalized_landmarks[fingertips[j]]
                )
                distances.append(dist)
        
        # 組合特徵
        features = np.concatenate([
            normalized_landmarks.flatten(),  # 正規化後的座標 (21*3 = 63維)
            np.array(distances),            # 指尖距離 (10維)
            np.array([1.0 if handedness == 'Right' else 0.0])  # 手勢方向 (1維)
        ])
        
        return features

# 使用範例
def main():
    processor = HandLandmarkProcessor()
    
    # 假設您已經有了 HandLandmarkerResult
    def process_result(result: HandLandmarkerResult):
        # 1. 基本轉換
        basic_result = processor.convert_to_array(result)
        if basic_result:
            landmarks, handedness = basic_result
            print(f"Landmarks shape: {landmarks.shape}")
            print(f"Handedness: {handedness}")
            
        # 2. 提取特徵向量
        features = processor.extract_features(result)
        if features is not None:
            print(f"Feature vector shape: {features.shape}")
            
        return features

    # 在您的代碼中使用：
    """
    # 使用範例
    detector = mp.tasks.vision.HandLandmarker(...)
    image = mp.Image(...)
    result = detector.detect(image)
    features = process_result(result)
    """

if __name__ == "__main__":
    main()