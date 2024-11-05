import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class FaceVerificationModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 128,
        dropout_rate: float = 0.5
    ):
        """
        Face Verification 模型
        
        Args:
            in_channels (int): 輸入通道數，默認為3（RGB圖片）
            feature_dim (int): 最終特徵向量的維度
            dropout_rate (float): Dropout 率
        """
        super().__init__()
        
        # 第一個卷積塊
        self.conv11 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二個卷積塊
        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三個卷積塊
        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # 計算最後的特徵圖大小（假設輸入是 224x224）
        self.feature_size = 256 * (224 // (2**3)) * (224 // (2**3))
        
        # 全連接層
        self.fc1 = nn.Linear(14336, 512)
        self.fc2 = nn.Linear(256, feature_dim)
        
        self.dropout = nn.Dropout(dropout_rate)

        self.activation = SwiGLU()
        
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        單張圖片的前向傳播
        
        Args:
            x: 輸入圖片 tensor, shape [batch_size, channels, height, width]
            
        Returns:
            特徵向量 tensor, shape [batch_size, feature_dim]
        """
        # 第一個卷積塊
        x = self.activation(self.conv11(x))
        x = self.activation(self.conv12(x))
        x = self.bn1(x)
        x = self.pool1(x)
        
        # 第二個卷積塊
        x = self.activation(self.conv21(x))
        x = self.activation(self.conv22(x))
        x = self.bn2(x)
        x = self.pool2(x)
        
        # 第三個卷積塊
        x = self.activation(self.conv31(x))
        x = self.activation(self.conv32(x))
        x = self.bn3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全連接層
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # L2 正規化
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        一對圖片的前向傳播
        
        Args:
            x1: 第一張圖片 tensor, shape [batch_size, channels, height, width]
            x2: 第二張圖片 tensor, shape [batch_size, channels, height, width]
            
        Returns:
            特徵向量對 (feature1, feature2)
        """
        feature1 = self.forward_one(x1)
        feature2 = self.forward_one(x2)
        return feature1, feature2
    
    def predict(self, x1: torch.Tensor, x2: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        預測兩張圖片是否為同一個人
        
        Args:
            x1: 第一張圖片
            x2: 第二張圖片
            threshold: 相似度閾值
            
        Returns:
            預測結果 (0/1)
        """
        feature1, feature2 = self.forward(x1, x2)
        similarity = F.cosine_similarity(feature1, feature2)
        predictions = (similarity > threshold).float()
        return predictions

if __name__ == '__main__':
    # 測試模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 創建模型
    model = FaceVerificationModel(
        in_channels=3,
        feature_dim=128,
        dropout_rate=0.5
    ).to(device)
    
    # 生成測試數據
    batch_size = 4
    img_size = 224
    x1 = torch.randn(batch_size, 3, img_size, img_size).to(device)
    x2 = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    # 測試前向傳播
    print("\nTesting forward pass...")
    feature1, feature2 = model(x1, x2)
    print(f"Feature 1 shape: {feature1.shape}")
    print(f"Feature 2 shape: {feature2.shape}")
    
    # 測試相似度計算
    print("\nTesting similarity calculation...")
    similarity = F.cosine_similarity(feature1, feature2)
    print(f"Similarity shape: {similarity.shape}")
    print(f"Similarity values: {similarity.cpu().detach().numpy()}")
    
    # 測試預測
    print("\nTesting prediction...")
    predictions = model.predict(x1, x2)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions.cpu().detach().numpy()}")
    
    # 檢查模型參數
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params/1e6:.2f}M")
    
    # 測試模型在不同輸入大小的表現
    print("\nTesting different input sizes...")
    test_sizes = [160, 224, 256]
    for size in test_sizes:
        x1 = torch.randn(1, 3, size, size).to(device)
        x2 = torch.randn(1, 3, size, size).to(device)
        try:
            with torch.no_grad():
                feature1, feature2 = model(x1, x2)
            print(f"Successfully processed input size: {size}x{size}")
        except Exception as e:
            print(f"Failed to process input size: {size}x{size}")
            print(f"Error: {str(e)}")