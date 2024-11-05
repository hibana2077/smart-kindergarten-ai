import torch
from torch.utils.data import Dataset
from pathlib import Path
import random
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, List, Optional, Union

class FaceVerificationDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        split: str = 'train',
        img_size: Union[int, Tuple[int, int]] = 224,
        transform: Optional[transforms.Compose] = None,
        center_crop: Optional[Union[int, Tuple[int, int]]] = None,
        normalize: bool = True
    ):
        """
        Args:
            root_dir (str): 資料集根目錄路徑
            split (str): 'train' 或 'val'
            img_size (int or tuple): 目標圖片大小，可以是單一整數或 (height, width)
            transform: 自定義的轉換操作，若提供則會覆蓋預設轉換
            center_crop: 指定中心裁切大小，可選
            normalize: 是否要標準化圖片，預設使用 ImageNet 的均值和標準差
        """
        self.root_dir = Path(root_dir) / split
        
        # 設定預設的圖片轉換
        if transform is None:
            transform_list = []
            
            # 處理 resize
            if isinstance(img_size, int):
                img_size = (img_size, img_size)
                
            # 先 resize 到稍大的尺寸（如果有指定 center_crop）
            if center_crop:
                if isinstance(center_crop, int):
                    center_crop = (center_crop, center_crop)
                transform_list.extend([
                    transforms.Resize((int(center_crop[0] * 1.1), int(center_crop[1] * 1.1))),
                    transforms.CenterCrop(center_crop)
                ])
            
            # 一般 resize
            transform_list.append(transforms.Resize(img_size))
            
            # 轉換為 tensor
            transform_list.append(transforms.ToTensor())
            
            # 標準化
            if normalize:
                transform_list.append(
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                )
            
            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = transform
        
        # 取得所有人的資料夾
        self.person_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        
        # 為每個人建立圖片路徑列表
        self.person_images = {}
        for person_dir in self.person_dirs:
            person_id = person_dir.name
            self.person_images[person_id] = list(person_dir.glob('*.[jp][pn][gf]'))
            
        # 建立所有可能的配對
        self.pairs = self._generate_pairs()
        
    def _generate_pairs(self) -> List[Tuple[str, str, int]]:
        """生成訓練用的圖片對
        Returns:
            List of (img1_path, img2_path, label)
            label: 1 表示同一人, 0 表示不同人
        """
        pairs = []
        
        # 生成正樣本 (同一人的圖片對)
        for person_id, images in self.person_images.items():
            if len(images) < 2:
                continue
            # 從每個人的圖片中取出所有可能的配對
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    pairs.append((str(images[i]), str(images[j]), 1))
                    
        # 生成負樣本 (不同人的圖片對)
        num_positive = len(pairs)  # 保持正負樣本平衡
        person_ids = list(self.person_images.keys())
        
        while len(pairs) < 2 * num_positive:
            # 隨機選擇兩個不同的人
            person1, person2 = random.sample(person_ids, 2)
            img1 = random.choice(self.person_images[person1])
            img2 = random.choice(self.person_images[person2])
            pairs.append((str(img1), str(img2), 0))
            
        random.shuffle(pairs)
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img1_path, img2_path, label = self.pairs[idx]
        
        # 讀取並轉換圖片
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, torch.tensor(label, dtype=torch.float32)