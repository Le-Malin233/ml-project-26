import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

class SketchTransformer(nn.Module):
    """基于坐标序列的草图识别Transformer模型"""
    
    def __init__(self, num_classes=345):  # 移除Config依赖
        super().__init__()
        
        # 在函数内部获取配置
        from config import get_config
        config = get_config()
        dropout_rate = config.get('DROPOUT', 0.3)
        
        self.d_model = 256  # 增加模型维度，提高表达能力
        self.nhead = 8
        self.num_layers = 6  # 增加层数
        
        # 输入嵌入层 - 添加非线性增强特征提取
        self.input_embedding = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate/3),
            nn.Linear(64, self.d_model)
        )
        
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Transformer编码器 - 使用GELU和pre-norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=1024,  # 增加前馈网络维度
            dropout=dropout_rate,
            batch_first=True,
            activation='gelu',  # 使用GELU激活函数，训练更稳定
            norm_first=True  # 使用pre-norm，训练更稳定
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model)  # 添加最终的层归一化
        )
        
        # 全局池化 - 使用多种池化方式结合
        self.attention_pool = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Tanh()
        )
        
        # 分类器 - 添加层归一化和更强的非线性
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),  # 添加层归一化，稳定训练
            nn.Linear(self.d_model, 512),
            nn.GELU(),  # 使用GELU
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x形状: (batch_size, seq_len=100, features=2)
        
        # 嵌入和位置编码
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 注意力池化 + 平均池化结合
        attention_weights = F.softmax(self.attention_pool(x), dim=1)
        attention_pooled = torch.sum(attention_weights * x, dim=1)
        mean_pooled = torch.mean(x, dim=1)
        
        # 结合两种池化结果
        x = attention_pooled + mean_pooled
        
        # 分类
        output = self.classifier(x)
        return output