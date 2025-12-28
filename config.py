# config.py
import torch

# 使用字典而不是类，避免导入问题
CONFIG = {
    # 数据路径
    'DATA_ROOT': "./data",
    'COORDINATE_PATH': "./data/coordinate_files",
    'PICTURE_PATH': "./data/picture_files",
    
    # 训练参数
    'BATCH_SIZE': 64,
    'LEARNING_RATE': 0.001,
    'EPOCHS': 50,
    'NUM_CLASSES': 345,
    
    # 模型参数
    'HIDDEN_SIZE': 256,
    'NUM_LAYERS': 2,
    'DROPOUT': 0.3,
    
    # 设备
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    
    # 保存路径
    'SAVE_DIR': "./checkpoints",
    'LOG_DIR': "./logs"
}

# 为了方便访问，也可以创建getter函数
def get_config():
    return CONFIG

def update_config(**kwargs):
    """更新配置"""
    CONFIG.update(kwargs)