# train.py 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import argparse

# 导入模块
from utils.data_loader import get_data_loaders, SketchDataset
from models.sketch_cnn import SketchCNN
from models.sketch_rnn import SketchRNN

class Config:
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 50
    NUM_CLASSES = 345
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR = "./checkpoints"
    LOG_DIR = "./logs"
    DATA_ROOT = "./data"

class Trainer:
    def __init__(self, model_type="cnn", batch_size=64, lr=0.001, epochs=50):
        self.model_type = model_type
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = Config.DEVICE
        
        # 创建目录
        os.makedirs(Config.SAVE_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        
        # 初始化模型
        self.model = self._init_model()
        self.model.to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 数据加载器
        print("加载数据...")
        self.data_loaders = get_data_loaders(
            batch_size=self.batch_size, 
            data_root=Config.DATA_ROOT
        )
        
        # TensorBoard
        self.writer = SummaryWriter(Config.LOG_DIR)
        
        print(f"使用设备: {self.device}")
        print(f"模型类型: {model_type}")
        print(f"批大小: {batch_size}, 学习率: {lr}, 轮数: {epochs}")
    
    def _init_model(self):
        if self.model_type == "cnn":
            return SketchCNN(num_classes=Config.NUM_CLASSES)
        elif self.model_type == "rnn":
            return SketchRNN(num_classes=Config.NUM_CLASSES)

        else:
            raise ValueError(f"未知模型类型: {self.model_type}")
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        if self.model_type == "cnn":
            train_loader = self.data_loaders['image']['train']
        else:
            train_loader = self.data_loaders['coordinate']['train']
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        self.writer.add_scalar('Train/Loss', avg_loss, epoch)
        self.writer.add_scalar('Train/Accuracy', accuracy, epoch)
        
        return avg_loss, accuracy
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        if self.model_type == "cnn":
            val_loader = self.data_loaders['image']['val']
        else:
            val_loader = self.data_loaders['coordinate']['val']
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', accuracy, epoch)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, accuracy, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'model_type': self.model_type,
            'batch_size': self.batch_size,
            'lr': self.lr
        }
        
        filename = f"{self.model_type}_epoch_{epoch}_acc_{accuracy:.2f}.pth"
        save_path = os.path.join(Config.SAVE_DIR, filename)
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = os.path.join(Config.SAVE_DIR, f"{self.model_type}_best.pth")
            torch.save(checkpoint, best_path)
    
    def train(self):
        print(f"开始训练 {self.model_type.upper()} 模型...")
        
        best_val_acc = 0
        
        for epoch in range(self.epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
            
            # 验证
            val_loss, val_acc = self.validate(epoch)
            print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 保存检查点
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                print(f"新的最佳验证准确率: {best_val_acc:.2f}%")
            
            if (epoch + 1) % 5 == 0 or is_best:
                self.save_checkpoint(epoch, val_acc, is_best)
        
        self.writer.close()
        print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
        
        # 保存最终模型
        final_path = os.path.join(Config.SAVE_DIR, f"{self.model_type}_final.pth")
        torch.save(self.model.state_dict(), final_path)
        print(f"最终模型已保存: {final_path}")

def main():
    parser = argparse.ArgumentParser(description='草图识别模型训练')
    parser.add_argument('--model', type=str, default='cnn', 
                       choices=['cnn', 'rnn'],
                       help='选择模型类型: cnn, rnn')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    
    args = parser.parse_args()
    
    # 开始训练
    trainer = Trainer(
        model_type=args.model,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs
    )
    trainer.train()

if __name__ == "__main__":

    main()
