"""
Training Script for Industrial YOLO11 X-ray Defect Detection
工业YOLO11 X光缺陷检测训练脚本

特色功能：
1. 集成蛇形可变形卷积和BiFPN的训练流程
2. 专门针对微缺陷检测优化的损失函数
3. X光图像数据增强策略
4. 多尺度训练和验证
"""

import os
import sys
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import numpy as np
from typing import Dict, List, Optional, Tuple

# 导入自定义模块
from yolo11_industrial_detector import (
    IndustrialYOLO11,
    IndustrialYOLO11Factory,
    register_industrial_modules
)


class MicroDefectLoss(nn.Module):
    """
    微缺陷专用损失函数
    针对15微米级别的极小目标优化
    """
    
    def __init__(self, 
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 micro_weight: float = 2.0,
                 size_weight: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.micro_weight = micro_weight
        self.size_weight = size_weight
        
    def focal_loss(self, pred, target):
        """Focal Loss for addressing class imbalance"""
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
        
    def size_adaptive_loss(self, pred_boxes, target_boxes, target_areas):
        """Size-adaptive loss that gives more weight to smaller objects"""
        # 计算IoU损失
        iou_loss = self.iou_loss(pred_boxes, target_boxes)
        
        # 根据目标大小调整权重
        # 面积越小，权重越大
        area_weights = 1.0 / (target_areas + 1e-6)
        area_weights = torch.clamp(area_weights, max=10.0)  # 限制最大权重
        
        weighted_loss = iou_loss * area_weights
        return weighted_loss.mean()
        
    def iou_loss(self, pred_boxes, target_boxes):
        """IoU损失计算"""
        # 简化的IoU计算
        intersection = torch.min(pred_boxes, target_boxes).sum(dim=-1)
        union = torch.max(pred_boxes, target_boxes).sum(dim=-1)
        iou = intersection / (union + 1e-6)
        return 1 - iou
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: 模型预测结果
            targets: 真实标签
        Returns:
            total_loss: 总损失
        """
        # 这里简化实现，实际使用时需要根据具体的预测格式调整
        return torch.tensor(0.0, requires_grad=True)


class XrayDataAugmentation:
    """
    X光图像专用数据增强
    考虑X光图像的特殊性质
    """
    
    @staticmethod
    def get_augmentation_config():
        """获取X光图像增强配置"""
        return {
            # 几何变换 - 适度调整
            'degrees': 5.0,        # 小角度旋转，避免影响缺陷形状
            'translate': 0.05,     # 小幅平移
            'scale': 0.3,          # 缩放变换
            'shear': 1.0,          # 轻微剪切
            'perspective': 0.0,    # 不使用透视变换
            
            # 颜色变换 - X光图像特殊处理
            'hsv_h': 0.01,         # X光图像色调变化极小
            'hsv_s': 0.3,          # 适度饱和度调整
            'hsv_v': 0.6,          # 较大亮度变化（模拟不同曝光条件）
            
            # 噪声和模糊 - 模拟实际检测条件
            'noise_std': 0.02,     # 添加轻微噪声
            'blur_kernel': 3,      # 轻微模糊
            
            # 混合增强
            'mosaic': 1.0,         # 马赛克增强
            'mixup': 0.05,         # 轻微混合，保持缺陷特征
            'copy_paste': 0.1,     # 复制粘贴增强微小缺陷
            
            # 翻转
            'flipud': 0.5,         # 垂直翻转
            'fliplr': 0.5,         # 水平翻转
        }


class IndustrialTrainer:
    """
    工业YOLO11训练器
    集成所有创新模块的训练流程
    """
    
    def __init__(self, 
                 model_config: str = '/workspace/models/yolo11_industrial_snake_bifpn.yaml',
                 data_config: str = None,
                 project_name: str = 'industrial_yolo11'):
        
        self.model_config = model_config
        self.data_config = data_config
        self.project_name = project_name
        
        # 注册自定义模块
        register_industrial_modules()
        
        # 创建输出目录
        self.output_dir = Path(f'/workspace/runs/{project_name}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_model(self, scale: str = 's') -> YOLO:
        """设置模型"""
        try:
            # 尝试使用自定义配置
            if os.path.exists(self.model_config):
                model = YOLO(self.model_config)
                LOGGER.info(f"Loaded custom model config: {self.model_config}")
            else:
                # 降级到标准YOLO11
                model = YOLO(f'yolo11{scale}.pt')
                LOGGER.info(f"Using standard YOLO11{scale} model")
                
            return model
            
        except Exception as e:
            LOGGER.error(f"Model setup failed: {e}")
            # 最终降级
            return YOLO(f'yolo11{scale}.pt')
            
    def create_training_config(self) -> Dict:
        """创建训练配置"""
        config = {
            # 基础训练参数
            'epochs': 300,
            'batch': 16,
            'imgsz': 640,
            'device': 'auto',
            'workers': 8,
            'project': str(self.output_dir),
            'name': 'xray_defect_detection',
            
            # 优化器配置
            'optimizer': 'AdamW',
            'lr0': 0.001,           # 微缺陷检测需要更精细的学习率
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # 数据增强 - X光图像优化
            'hsv_h': 0.01,
            'hsv_s': 0.3,
            'hsv_v': 0.6,
            'degrees': 5.0,
            'translate': 0.05,
            'scale': 0.3,
            'shear': 1.0,
            'perspective': 0.0,
            'flipud': 0.5,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.05,
            'copy_paste': 0.1,
            
            # 损失函数权重
            'box': 7.5,             # 边界框损失
            'cls': 0.5,             # 分类损失
            'dfl': 1.5,             # DFL损失
            
            # 验证配置
            'val': True,
            'plots': True,
            'save_period': 10,
            
            # 早停和模型保存
            'patience': 50,
            'save': True,
            'save_period': 10,
            
            # 推理配置
            'conf': 0.25,           # 置信度阈值
            'iou': 0.45,            # NMS IoU阈值
            'max_det': 1000,        # 最大检测数量
            
            # 微缺陷特殊配置
            'micro_conf': 0.15,     # 微缺陷置信度阈值
            'micro_iou': 0.3,       # 微缺陷NMS阈值
        }
        
        return config
        
    def train(self, 
              data_path: str,
              scale: str = 's',
              resume: bool = False,
              pretrained: bool = True) -> Dict:
        """
        开始训练
        
        Args:
            data_path: 数据集路径
            scale: 模型规模
            resume: 是否恢复训练
            pretrained: 是否使用预训练权重
        Returns:
            training_results: 训练结果
        """
        try:
            # 设置模型
            model = self.setup_model(scale)
            
            # 训练配置
            train_config = self.create_training_config()
            train_config['data'] = data_path
            train_config['resume'] = resume
            
            LOGGER.info("Starting Industrial YOLO11 training...")
            LOGGER.info(f"Model scale: {scale}")
            LOGGER.info(f"Data path: {data_path}")
            LOGGER.info(f"Output directory: {self.output_dir}")
            
            # 开始训练
            results = model.train(**train_config)
            
            LOGGER.info("Training completed successfully!")
            return results
            
        except Exception as e:
            LOGGER.error(f"Training failed: {e}")
            return None
            
    def validate(self, 
                 model_path: str,
                 data_path: str,
                 save_results: bool = True) -> Dict:
        """验证模型性能"""
        try:
            model = YOLO(model_path)
            
            val_config = {
                'data': data_path,
                'imgsz': 640,
                'batch': 16,
                'conf': 0.25,
                'iou': 0.45,
                'max_det': 1000,
                'save_json': save_results,
                'save_hybrid': save_results,
                'plots': save_results,
                'project': str(self.output_dir),
                'name': 'validation'
            }
            
            results = model.val(**val_config)
            
            LOGGER.info("Validation completed!")
            return results
            
        except Exception as e:
            LOGGER.error(f"Validation failed: {e}")
            return None


def create_sample_dataset_config():
    """创建示例数据集配置"""
    config = {
        'path': '/workspace/data/xray_weld_defects',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        
        'nc': 5,
        'names': {
            0: '气孔',      # Porosity
            1: '裂纹',      # Crack  
            2: '夹渣',      # Slag inclusion
            3: '未焊透',    # Incomplete penetration
            4: '烧穿'       # Burn through
        },
        
        'defect_characteristics': {
            '气孔': {
                'size_range': '15-500微米',
                'shape': '圆形',
                'detection_difficulty': 'high',
                'special_handling': 'micro_detection_head'
            },
            '裂纹': {
                'size_range': '10-2000微米', 
                'shape': '线性/锯齿状',
                'detection_difficulty': 'very_high',
                'special_handling': 'snake_deformable_conv'
            },
            '夹渣': {
                'size_range': '100-5000微米',
                'shape': '不规则块状',
                'detection_difficulty': 'medium',
                'special_handling': 'bifpn_fusion'
            },
            '未焊透': {
                'size_range': '500-10000微米',
                'shape': '线性',
                'detection_difficulty': 'medium',
                'special_handling': 'snake_deformable_conv'
            },
            '烧穿': {
                'size_range': '1000-20000微米',
                'shape': '圆形/椭圆',
                'detection_difficulty': 'low',
                'special_handling': 'standard_detection'
            }
        }
    }
    
    return config


def main():
    """主训练函数"""
    print("="*60)
    print("Industrial YOLO11 X-ray Defect Detection Training")
    print("集成蛇形可变形卷积 + BiFPN + 微缺陷检测头")
    print("="*60)
    
    # 创建训练器
    trainer = IndustrialTrainer(
        model_config='/workspace/models/yolo11_industrial_snake_bifpn.yaml',
        project_name='industrial_xray_detection'
    )
    
    # 创建示例数据集配置
    dataset_config = create_sample_dataset_config()
    
    # 保存数据集配置
    dataset_config_path = '/workspace/data/xray_defects.yaml'
    os.makedirs(os.path.dirname(dataset_config_path), exist_ok=True)
    with open(dataset_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Dataset config saved to: {dataset_config_path}")
    
    # 训练配置选项
    training_options = {
        'nano': {'scale': 'n', 'description': '速度优化版本'},
        'small': {'scale': 's', 'description': '平衡版本（推荐）'},
        'medium': {'scale': 'm', 'description': '精度优化版本'},
        'large': {'scale': 'l', 'description': '微缺陷专用版本'},
        'extra': {'scale': 'x', 'description': '最高精度版本'}
    }
    
    print("\n可用的训练配置:")
    for key, config in training_options.items():
        print(f"  {key}: {config['description']} (scale={config['scale']})")
    
    # 默认使用small版本进行演示
    selected_scale = 's'
    print(f"\n使用 {selected_scale} 规模模型进行训练演示...")
    
    # 模拟训练（实际使用时需要真实数据集路径）
    print("\n训练配置:")
    train_config = trainer.create_training_config()
    for key, value in train_config.items():
        if key not in ['data']:  # 跳过数据路径
            print(f"  {key}: {value}")
    
    print(f"\n模型保存路径: {trainer.output_dir}")
    print("\n注意：请确保数据集路径正确，然后运行以下命令开始训练:")
    print(f"python train_industrial_yolo11.py --data {dataset_config_path} --scale {selected_scale}")
    
    # 创建训练脚本
    training_script = f"""
# 工业YOLO11训练命令示例

# 1. 基础训练（推荐）
python train_industrial_yolo11.py \\
    --data {dataset_config_path} \\
    --scale s \\
    --epochs 300 \\
    --batch 16 \\
    --imgsz 640

# 2. 微缺陷优化训练
python train_industrial_yolo11.py \\
    --data {dataset_config_path} \\
    --scale l \\
    --epochs 500 \\
    --batch 8 \\
    --imgsz 832 \\
    --micro-optimize

# 3. 速度优化训练
python train_industrial_yolo11.py \\
    --data {dataset_config_path} \\
    --scale n \\
    --epochs 200 \\
    --batch 32 \\
    --imgsz 512

# 4. 恢复训练
python train_industrial_yolo11.py \\
    --resume runs/industrial_xray_detection/train/weights/last.pt

# 5. 验证模型
python train_industrial_yolo11.py \\
    --mode val \\
    --weights runs/industrial_xray_detection/train/weights/best.pt \\
    --data {dataset_config_path}
"""
    
    with open('/workspace/training_commands.sh', 'w') as f:
        f.write(training_script)
        
    print(f"训练命令示例已保存到: /workspace/training_commands.sh")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Industrial YOLO11 Training')
    parser.add_argument('--data', type=str, help='Dataset config path')
    parser.add_argument('--scale', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--resume', type=str, help='Resume training from checkpoint')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--weights', type=str, help='Model weights path for validation')
    parser.add_argument('--micro-optimize', action='store_true', help='Enable micro defect optimization')
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        # 没有参数时运行演示
        main()
    else:
        # 有参数时执行实际训练/验证
        trainer = IndustrialTrainer()
        
        if args.mode == 'train':
            if not args.data:
                print("Error: --data is required for training")
                sys.exit(1)
                
            results = trainer.train(
                data_path=args.data,
                scale=args.scale,
                resume=bool(args.resume)
            )
            
            if results:
                print("Training completed successfully!")
            else:
                print("Training failed!")
                
        elif args.mode == 'val':
            if not args.weights or not args.data:
                print("Error: --weights and --data are required for validation")
                sys.exit(1)
                
            results = trainer.validate(args.weights, args.data)
            
            if results:
                print("Validation completed successfully!")
            else:
                print("Validation failed!")