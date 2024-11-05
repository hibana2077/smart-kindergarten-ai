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
        """生成合成測試數據"""
        synthetic_images = []
        ground_truth = []
        
        for i in tqdm(range(num_samples)):
            # 創建空白畫布
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            image.fill(255)
            
            # 隨機選擇形狀
            shape_type = np.random.choice(['circle', 'square', 'triangle'])
            
            # 添加隨機變形和噪聲
            distortion = np.random.uniform(0, 0.3)  # 變形程度
            noise_level = np.random.uniform(0, 0.1)  # 噪聲程度
            
            # 生成形狀
            center = (np.random.randint(200, 300), np.random.randint(200, 300))
            size = np.random.randint(50, 150)
            
            if shape_type == 'circle':
                cv2.circle(image, center, size, (0, 0, 0), 2)
                if distortion > 0:
                    # 添加橢圓變形
                    cv2.ellipse(image, center, (size, int(size * (1 + distortion))), 
                              0, 0, 360, (0, 0, 0), 2)
            elif shape_type == 'square':
                points = np.array([
                    [center[0] - size, center[1] - size],
                    [center[0] + size, center[1] - size],
                    [center[0] + size, center[1] + size],
                    [center[0] - size, center[1] + size]
                ])
                if distortion > 0:
                    # 添加透視變形
                    points += np.random.normal(0, size * distortion, points.shape)
                cv2.polylines(image, [points.astype(np.int32)], True, (0, 0, 0), 2)
            else:  # triangle
                height = int(size * 1.732)  # 等邊三角形高度
                points = np.array([
                    [center[0], center[1] - height//2],
                    [center[0] - size, center[1] + height//2],
                    [center[0] + size, center[1] + height//2]
                ])
                if distortion > 0:
                    # 添加不規則變形
                    points += np.random.normal(0, size * distortion, points.shape)
                cv2.polylines(image, [points.astype(np.int32)], True, (0, 0, 0), 2)
            
            # 添加噪聲
            if noise_level > 0:
                noise = np.random.normal(0, noise_level * 255, image.shape)
                image = np.clip(image + noise, 0, 255).astype(np.uint8)
            
            synthetic_images.append(image)
            ground_truth.append({
                'shape_type': shape_type,
                'distortion': distortion,
                'noise_level': noise_level,
                'size': size
            })
            
            # 保存生成的圖片
            cv2.imwrite(f"{self.results_dir}/images/synthetic_{i}.png", image)
            
        return synthetic_images, ground_truth
    
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
        plt.style.use('seaborn')
        
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