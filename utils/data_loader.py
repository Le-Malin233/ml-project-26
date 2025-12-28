# utils/data_loader.py
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

def create_label_mapping(data_root):
    """创建类别标签映射"""
    train_path = os.path.join(data_root, "coordinate_files/train")
    categories = sorted(os.listdir(train_path))
    label_to_idx = {cat: i for i, cat in enumerate(categories)}
    idx_to_label = {i: cat for i, cat in enumerate(categories)}
    
    # 保存映射文件
    os.makedirs("mappings", exist_ok=True)
    with open("mappings/label_mapping.json", "w") as f:
        json.dump(label_to_idx, f)
    
    return label_to_idx, idx_to_label

class SketchDataset(Dataset):
    """通用草图数据集类，支持序列和图像数据"""
    
    def __init__(self, data_type="coordinate", split="train", transform=None, data_root="./data"):
        """
        Args:
            data_type: "coordinate" 或 "image"
            split: "train", "val", "test"
            transform: 图像变换
            data_root: 数据根目录
        """
        super().__init__()
        
        self.data_root = data_root
        
        if data_type == "coordinate":
            self.data_path = os.path.join(data_root, "coordinate_files")
            self.file_ext = ".npy"
        else:  # image
            self.data_path = os.path.join(data_root, "picture_files")
            self.file_ext = ".png"
            self.transform = transforms.Compose([
            transforms.Resize((112, 112)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.data_type = data_type
        self.split = split
        self.samples = []
        self.labels = []
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载数据路径和标签"""
        split_path = os.path.join(self.data_path, self.split)
        
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"路径不存在: {split_path}")
        
        # 获取类别标签映射
        self.label_to_idx, _ = create_label_mapping(self.data_root)
        
        # 遍历所有类别文件夹
        for category in os.listdir(split_path):
            if category not in self.label_to_idx:
                continue
                
            category_path = os.path.join(split_path, category)
            if not os.path.isdir(category_path):
                continue
            
            label = self.label_to_idx[category]
            
            # 获取该类别所有文件
            for file_name in os.listdir(category_path):
                if file_name.endswith(self.file_ext):
                    file_path = os.path.join(category_path, file_name)
                    self.samples.append(file_path)
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path = self.samples[idx]
        label = self.labels[idx]
        
        if self.data_type == "coordinate":
            # 加载坐标序列数据
            try:
                data = np.load(file_path, encoding='latin1', allow_pickle=True)
            except:
                # 如果加载失败，返回零数组
                data = np.zeros((100, 4), dtype='float32')
            
            if data.dtype == 'object':
                data = data[0]
            
            # 统一形状为 (100, 2)
            if data.shape[0] > 100:
                data = data[:100]
            elif data.shape[0] < 100:
                # 填充到100
                padding = np.zeros((100 - data.shape[0], data.shape[1]))
                data = np.vstack([data, padding])
            
            if data.shape[1] >= 2:
                data = data[:, :2].astype('float32')
            else:
                data = np.zeros((100, 2), dtype='float32')
            
            # 标准化坐标
            if np.max(data) > 0:
                data = data / np.max(data)
            
            return torch.FloatTensor(data), label
            
        else:  # image
            # 加载图像数据
            try:
                image = Image.open(file_path).convert('L')  # 转为灰度图
                if self.transform:
                    image = self.transform(image)
                return image, label
            except:
                # 如果图像加载失败，返回黑色图像
                black_image = torch.zeros((1, 224, 224))
                return black_image, label

def get_data_loaders(batch_size=64, data_root="./data"):
    """获取训练、验证、测试数据加载器"""
    
    # 图像数据加载器
    train_image_dataset = SketchDataset(data_type="image", split="train", data_root=data_root)
    val_image_dataset = SketchDataset(data_type="image", split="val", data_root=data_root)
    test_image_dataset = SketchDataset(data_type="image", split="test", data_root=data_root)
    
    train_image_loader = DataLoader(
        train_image_dataset, batch_size=batch_size, shuffle=True, num_workers=0  # Windows上设置为0
    )
    val_image_loader = DataLoader(
        val_image_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_image_loader = DataLoader(
        test_image_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # 坐标序列数据加载器
    train_coord_dataset = SketchDataset(data_type="coordinate", split="train", data_root=data_root)
    val_coord_dataset = SketchDataset(data_type="coordinate", split="val", data_root=data_root)
    test_coord_dataset = SketchDataset(data_type="coordinate", split="test", data_root=data_root)
    
    train_coord_loader = DataLoader(
        train_coord_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_coord_loader = DataLoader(
        val_coord_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_coord_loader = DataLoader(
        test_coord_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return {
        'image': {
            'train': train_image_loader,
            'val': val_image_loader,
            'test': test_image_loader
        },
        'coordinate': {
            'train': train_coord_loader,
            'val': val_coord_loader,
            'test': test_coord_loader
        }
    }