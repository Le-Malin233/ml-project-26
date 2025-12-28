# models/sketch_rnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SketchRNN(nn.Module):
    """基于坐标序列的草图识别RNN模型"""
    
    def __init__(self, num_classes=345):  # 移除Config依赖
        super().__init__()
        
        # 在函数内部获取配置
        from config import get_config
        config = get_config()
        
        hidden_size = config.get('HIDDEN_SIZE', 256)
        num_layers = config.get('NUM_LAYERS', 2)
        dropout_rate = config.get('DROPOUT', 0.3)
        
        # 输入特征: (x, y) 坐标
        input_size = 2
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x形状: (batch_size, seq_len=100, features=2)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 注意力权重
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 加权求和
        weighted = torch.sum(lstm_out * attention_weights, dim=1)
        
        # 分类
        output = self.classifier(weighted)
        return output