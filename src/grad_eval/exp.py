import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from geometry_assessment import GeometryAssessment  # 引入先前定義的類
import os

class GeometryExperiment:
    def __init__(self):
        self.assessor = GeometryAssessment()
        self.results_dir = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/images", exist_ok=True)
        
    def generate_synthetic_data(self, num_samples=100):
        """生成更有變化的合成測試數據"""
        synthetic_images = []
        ground_truth = []
        
        for i in tqdm(range(num_samples)):
            # 創建基礎畫布，隨機背景色以模擬紙張顏色變化
            bg_color = np.random.randint(240, 256)
            image = np.ones((512, 512, 3), dtype=np.uint8) * bg_color
            
            # 隨機選擇形狀，增加權重使某些形狀出現機率較高
            shape_weights = [0.4, 0.35, 0.25]  # 圓形、方形、三角形的權重
            shape_type = np.random.choice(['circle', 'square', 'triangle'], p=shape_weights)
            
            # 更有變化的變形和噪聲參數
            # 使用beta分布來生成偏向小值但偶爾會有大值的參數
            distortion = np.random.beta(2, 5) * 0.8  # 增加變形範圍，使用beta分布
            noise_level = np.random.beta(2, 7) * 0.4  # 增加噪聲範圍
            rotation = np.random.normal(0, 30)  # 正態分布的旋轉角度
            
            # 品質因子使用beta分布，使其有較大變異性
            quality_factor = np.random.beta(5, 2) * 0.7 + 0.3  # 0.3 到 1.0 之間
            
            # 更多變化的位置和大小
            # 使用正態分布使形狀更可能出現在中心附近
            center_x = np.random.normal(256, 50)
            center_y = np.random.normal(256, 50)
            center = (int(np.clip(center_x, 100, 412)), 
                     int(np.clip(center_y, 100, 412)))
            
            # 大小使用對數正態分布，使其有時會出現特別大或小的形狀
            size = int(np.random.lognormal(mean=4, sigma=0.3))
            size = np.clip(size, 30, 200)
            
            # 生成形狀點
            points = self._generate_shape_points(shape_type, center, size, quality_factor)
            
            # 添加非線性變形
            if distortion > 0:
                points = self._apply_complex_distortion(points, distortion, shape_type)
            
            # 添加旋轉，使用正態分布的角度
            points = self._apply_rotation(points, center, rotation)
            
            # 添加手繪效果
            points = self._add_hand_drawn_effect(points, quality_factor)
            
            # 繪製形狀，線條粗細隨機
            thickness = np.random.randint(1, 4)
            self._draw_shape(image, points, shape_type, thickness)
            
            # 添加更複雜的噪聲效果
            if noise_level > 0:
                image = self._add_complex_noise(image, noise_level)
            
            # 隨機調整對比度和亮度
            image = self._adjust_image_properties(image)
            
            # 記錄真實值和品質資訊
            quality_metrics = {
                'shape_type': shape_type,
                'distortion': distortion,
                'noise_level': noise_level,
                'rotation': rotation,
                'quality_factor': quality_factor,
                'size': size,
                'expected_score': self._calculate_expected_score(
                    distortion, noise_level, quality_factor
                )
            }
                
            synthetic_images.append(image)
            ground_truth.append(quality_metrics)
            
            # 保存生成的圖片
            cv2.imwrite(f"{self.results_dir}/images/synthetic_{i}.png", image)
            
        return synthetic_images, ground_truth
    
    def _add_complex_noise(self, image, noise_level):
        """添加更複雜的噪聲效果"""
        # 高斯噪聲
        noise = np.random.normal(0, noise_level * 50, image.shape)
        noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # 隨機添加椒鹽噪聲
        if np.random.random() < 0.3:
            prob = noise_level * 0.1
            salt = np.random.random(image.shape[:2]) < prob
            pepper = np.random.random(image.shape[:2]) < prob
            noisy[salt] = 255
            noisy[pepper] = 0
        
        # 隨機模糊
        if np.random.random() < 0.4:
            kernel_size = np.random.choice([3, 5, 7])
            sigma = np.random.uniform(0.5, 2.0)
            noisy = cv2.GaussianBlur(noisy, (kernel_size, kernel_size), sigma)
        
        return noisy

    def _add_hand_drawn_effect(self, points, quality_factor):
        """添加手繪效果"""
        # 根據品質因子添加抖動
        jitter = (1 - quality_factor) * 5
        noise = np.random.normal(0, jitter, points.shape)
        points = points + noise
        
        # 隨機調整點的位置模擬手繪不平整
        if np.random.random() < 0.3:
            points += np.random.normal(0, 2, points.shape)
        
        return points

    def _adjust_image_properties(self, image):
        """隨機調整圖像屬性"""
        # 隨機調整對比度
        contrast = np.random.uniform(0.8, 1.2)
        image = np.clip(image * contrast, 0, 255).astype(np.uint8)
        
        # 隨機調整亮度
        brightness = np.random.randint(-20, 21)
        image = np.clip(image + brightness, 0, 255).astype(np.uint8)
        
        return image

    def _apply_complex_distortion(self, points, distortion, shape_type):
        """應用更複雜的非線性變形"""
        if shape_type == 'circle':
            # 橢圓變形加上波紋效果
            angles = np.arctan2(points[:,1] - np.mean(points[:,1]), 
                              points[:,0] - np.mean(points[:,0]))
            radii = np.sqrt(np.sum((points - np.mean(points, axis=0))**2, axis=1))
            
            # 添加正弦波變形
            wave = np.sin(angles * np.random.randint(2, 5)) * distortion * 10
            radii += wave
            
            points[:,0] = np.mean(points[:,0]) + radii * np.cos(angles)
            points[:,1] = np.mean(points[:,1]) + radii * np.sin(angles)
            
        else:
            # 為其他形狀添加不規則變形
            center = np.mean(points, axis=0)
            for i in range(len(points)):
                # 計算到中心的距離
                dist = np.linalg.norm(points[i] - center)
                # 添加基於距離的非線性變形
                angle = np.random.uniform(0, 2*np.pi)
                offset = dist * distortion * np.random.uniform(0.5, 1.5)
                points[i] += np.array([np.cos(angle), np.sin(angle)]) * offset
        
        return points

    def _calculate_expected_score(self, distortion, noise_level, quality_factor):
        """計算預期分數"""
        # 基礎分數
        base_score = 90
        
        # 根據各因素扣分，使用非線性關係
        distortion_penalty = (distortion ** 1.5) * 40  # 非線性懲罰
        noise_penalty = noise_level * 25
        quality_penalty = ((1 - quality_factor) ** 1.2) * 15
        
        # 加入隨機變異，使用beta分布來產生更自然的變異
        random_variation = (np.random.beta(2, 5) - 0.5) * 10
        
        # 計算最終分數
        score = (base_score 
                - distortion_penalty 
                - noise_penalty 
                - quality_penalty 
                + random_variation)
        
        # 確保分數在合理範圍內，使用sigmoid函數使分數分布更自然
        score = 60 + 40 / (1 + np.exp(-0.1 * (score - 75)))
        
        return round(np.clip(score, 60, 100), 2)

    def _generate_shape_points(self, shape_type, center, size, quality_factor):
        """生成基礎形狀點"""
        cx, cy = center
        if shape_type == 'circle':
            # 生成圓形的點集
            angles = np.linspace(0, 2*np.pi, 32)
            radius = size * quality_factor
            points = np.array([[cx + radius*np.cos(a), cy + radius*np.sin(a)] 
                             for a in angles], dtype=np.float32)
            
        elif shape_type == 'square':
            # 生成方形的點集
            half_size = size * quality_factor / 2
            points = np.array([
                [cx - half_size, cy - half_size],
                [cx + half_size, cy - half_size],
                [cx + half_size, cy + half_size],
                [cx - half_size, cy + half_size]
            ], dtype=np.float32)
            
        else:  # triangle
            # 生成三角形的點集
            height = size * 1.732 * quality_factor
            half_size = size * quality_factor
            points = np.array([
                [cx, cy - height/2],
                [cx - half_size, cy + height/2],
                [cx + half_size, cy + height/2]
            ], dtype=np.float32)
            
        return points
    
    def _apply_distortion(self, points, distortion, shape_type):
        """應用非線性變形"""
        # 基於形狀類型的特定變形
        if shape_type == 'circle':
            # 橢圓變形
            points[:, 0] *= (1 + distortion * np.random.uniform(-0.5, 0.5))
            points[:, 1] *= (1 + distortion * np.random.uniform(-0.5, 0.5))
        else:
            # 點的隨機偏移
            noise = np.random.normal(0, distortion * 20, points.shape)
            points += noise
        return points
    
    def _apply_rotation(self, points, center, angle):
        """應用旋轉變換"""
        cx, cy = center
        rad = np.deg2rad(angle)
        cos_val = np.cos(rad)
        sin_val = np.sin(rad)
        
        # 移動到原點
        points = points - np.array([cx, cy])
        
        # 旋轉
        rotated_x = points[:, 0] * cos_val - points[:, 1] * sin_val
        rotated_y = points[:, 0] * sin_val + points[:, 1] * cos_val
        
        # 移回原位置
        points = np.column_stack((rotated_x, rotated_y)) + np.array([cx, cy])
        return points
    
    def _draw_shape(self, image, points, shape_type, thickness=2):
        """
        繪製形狀
        
        Parameters:
            image: 要繪製的圖像
            points: 形狀的頂點
            shape_type: 形狀類型
            thickness: 線條粗細，默認為2
        """
        points = points.astype(np.int32)
        
        # 確保 thickness 至少為 1
        thickness = max(1, thickness)
        
        # 隨機調整線條顏色，模擬不同筆觸
        color = np.random.randint(0, 50, size=3).tolist()  # 接近黑色但有細微變化
        
        if shape_type == 'circle':
            # 使用多邊形近似繪製圓形
            cv2.polylines(image, [points], True, color, thickness)
            
            # 隨機添加額外的細微線條來模擬手繪效果
            if np.random.random() < 0.3:
                for i in range(len(points)-1):
                    if np.random.random() < 0.2:  # 20%機率在某些段落添加重疊線條
                        overlay_thickness = max(1, thickness - 1)  # 確保至少為1
                        cv2.line(image, 
                                tuple(points[i]), 
                                tuple(points[i+1]), 
                                color,
                                overlay_thickness)
        else:
            # 直接繪製多邊形
            cv2.polylines(image, [points], True, color, thickness)
            
            # 對於方形和三角形，可能在角落處有重疊的線條
            if np.random.random() < 0.4:
                for i in range(len(points)):
                    if np.random.random() < 0.3:  # 30%機率在角落添加重疊
                        start_point = points[i]
                        end_point = points[(i+1) % len(points)]
                        
                        # 添加略微偏移的線條
                        offset = np.random.randint(-2, 3, size=2)
                        overlay_thickness = max(1, thickness - 1)  # 確保至少為1
                        cv2.line(image,
                                tuple(start_point + offset),
                                tuple(end_point + offset),
                                color,
                                overlay_thickness)
        
        # 可能在形狀周圍添加一些小的手繪痕跡
        if np.random.random() < 0.2:
            num_marks = np.random.randint(1, 4)
            for _ in range(num_marks):
                idx = np.random.randint(0, len(points))
                mark_length = np.random.randint(3, 8)
                mark_angle = np.random.uniform(0, 2*np.pi)
                
                start_point = points[idx]
                end_point = start_point + np.array([
                    int(mark_length * np.cos(mark_angle)),
                    int(mark_length * np.sin(mark_angle))
                ])
                
                # 使用更細的線條，但確保至少為1
                mark_thickness = max(1, thickness - 1)
                cv2.line(image,
                        tuple(start_point),
                        tuple(end_point),
                        color,
                        mark_thickness)
    
    def _add_noise_and_blur(self, image, noise_level):
        """添加噪聲和模糊效果"""
        # 高斯噪聲
        noise = np.random.normal(0, noise_level * 255, image.shape)
        noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # 隨機模糊
        if np.random.random() < 0.5:
            kernel_size = np.random.choice([3, 5, 7])
            noisy = cv2.GaussianBlur(noisy, (kernel_size, kernel_size), 0)
            
        return noisy
    
    def _calculate_expected_score(self, distortion, noise_level, quality_factor):
        """計算預期分數"""
        # 基礎分數
        base_score = 85
        
        # 根據各因素扣分
        distortion_penalty = distortion * 30  # 最多扣15分
        noise_penalty = noise_level * 20  # 最多扣4分
        quality_penalty = (1 - quality_factor) * 10  # 最多扣5分
        
        # 計算最終分數
        score = base_score - distortion_penalty - noise_penalty - quality_penalty
        
        # 確保分數在合理範圍內
        return np.clip(score, 60, 100)
    
    def run_experiment(self, num_samples=100):
        """執行實驗並收集結果"""
        print("生成合成測試數據...")
        images, ground_truth = self.generate_synthetic_data(num_samples)
        
        print("執行評估...")
        results = []
        for i, (image, truth) in enumerate(tqdm(zip(images, ground_truth))):
            assessment = self.assessor.assess_drawing(image)
            
            for shape_result in assessment:
                results.append({
                    'sample_id': i,
                    'true_shape': truth['shape_type'],
                    'detected_shape': shape_result['shape_type'],
                    'detection_confidence': shape_result['detection_confidence'],
                    'shape_similarity': shape_result['shape_similarity'],
                    'score': shape_result['score'],
                    'distortion': truth['distortion'],
                    'noise_level': truth['noise_level'],
                    'size': truth['size']
                })
        
        # 轉換為DataFrame
        df = pd.DataFrame(results)
        df.to_csv(f"{self.results_dir}/experiment_results.csv", index=False)
        return df
    
    def generate_visualizations(self, df):
        """生成可視化圖表"""
        # plt.style.use('seaborn')
        
        # 1. 準確率與變形程度的關係
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='distortion', y='score', hue='true_shape')
        plt.title('Score vs. Distortion Level by Shape Type')
        plt.xlabel('Distortion Level')
        plt.ylabel('Score')
        plt.savefig(f"{self.results_dir}/score_vs_distortion.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 混淆矩陣
        plt.figure(figsize=(8, 6))
        confusion = pd.crosstab(df['true_shape'], df['detected_shape'], normalize='index')
        sns.heatmap(confusion, annot=True, cmap='Blues', fmt='.2f')
        plt.title('Shape Detection Confusion Matrix')
        plt.savefig(f"{self.results_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 分數分布
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='true_shape', y='score')
        plt.title('Score Distribution by Shape Type')
        plt.xlabel('Shape Type')
        plt.ylabel('Score')
        plt.savefig(f"{self.results_dir}/score_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 噪聲影響分析
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='noise_level', y='detection_confidence', 
                       hue='true_shape', size='size', sizes=(20, 200))
        plt.title('Detection Confidence vs. Noise Level')
        plt.xlabel('Noise Level')
        plt.ylabel('Detection Confidence')
        plt.savefig(f"{self.results_dir}/noise_impact.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 性能指標摘要
        summary = df.groupby('true_shape').agg({
            'score': ['mean', 'std'],
            'detection_confidence': 'mean',
            'shape_similarity': 'mean'
        }).round(3)
        
        summary.to_csv(f"{self.results_dir}/performance_summary.csv")
        
        return summary

def main():
    # 設置隨機種子以確保可重複性
    np.random.seed(42)
    
    # 創建實驗實例
    experiment = GeometryExperiment()
    
    # 運行實驗
    print("開始運行實驗...")
    results_df = experiment.run_experiment(num_samples=100)
    
    # 生成視覺化結果
    print("生成視覺化結果...")
    summary = experiment.generate_visualizations(results_df)
    
    print(f"\n實驗結果已保存至: {experiment.results_dir}")
    print("\n性能摘要:")
    print(summary)

if __name__ == "__main__":
    main()