import os
import argparse

# IMPORTANT: Import custom modules BEFORE importing YOLO
print("📦 Loading industrial modules...")

# Import fixed industrial modules
import industrial_modules_fixed  # 修复版工业模块
import advanced_modules          # 现有的增强模块

# Now import YOLO after modules are registered
from ultralytics import YOLO
print("✅ YOLO imported successfully")


def parse_args():
    parser = argparse.ArgumentParser(description='Industrial X-ray Defect Detection - Fixed Version')
    parser.add_argument('--model', type=str, default='models/yolo11_industrial_working.yaml', help='Model config path')
    parser.add_argument('--data', type=str, default='data/xray_defects.yaml', help='Dataset config path')
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='Device (0, 1, cpu)')
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory')
    parser.add_argument('--name', type=str, default='industrial-xray-defect', help='Experiment name')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--weights', type=str, default='', help='Pretrained weights path')
    parser.add_argument('--micro-optimize', action='store_true', help='Enable micro defect optimization')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("🚀 Industrial X-ray Defect Detection Training")
    print("🔬 Four Core Innovations Implementation:")
    print("   1. 🔍 专用微缺陷检测头 (P1高分辨率层)")
    print("   2. 🐍 蛇形可变形卷积 (适应不规则缺陷)")
    print("   3. 🔄 双向三阶特征金字塔 (BiFPN)")
    print("   4. 📊 小目标优化损失函数")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    
    # 验证模型文件存在
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    # 创建模型
    print("🏗️ Creating industrial model...")
    try:
        model = YOLO(args.model)
        print("✅ Industrial model created successfully!")
        
        # 显示模型信息
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"📊 Model parameters: {total_params:,}")
        print(f"📊 Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Run: python debug_model.py")
        print("   2. Check: python ultimate_fix.py")
        return
    
    # 加载预训练权重
    if args.weights and os.path.exists(args.weights):
        model.load(args.weights)
        print(f"✅ Loaded pretrained weights: {args.weights}")

    # 小目标优化的训练超参数
    hyp = {
        'imgsz': args.imgsz,
        'batch': args.batch,
        'epochs': args.epochs,
        'device': args.device,
        'optimizer': 'AdamW',        # 更适合小目标
        'cos_lr': True,              # 余弦学习率
        'close_mosaic': 20,          # 最后20轮关闭mosaic
        
        # 学习率设置 - 针对微缺陷优化
        'lr0': 0.001,               # 较低的初始学习率
        'lrf': 0.01,                # 最终学习率因子
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # X光图像数据增强策略
        'hsv_h': 0.005,             # X光图像色调变化极小
        'hsv_s': 0.3,               # 适度饱和度调整
        'hsv_v': 0.5,               # 亮度变化（模拟不同曝光）
        'degrees': 3.0,             # 小角度旋转，保持缺陷形状
        'translate': 0.05,          # 小幅平移
        'scale': 0.4,               # 缩放增强
        'shear': 0.0,               # 不使用剪切，避免变形缺陷
        'perspective': 0.0,         # 不使用透视变换
        'flipud': 0.5,              # 垂直翻转
        'fliplr': 0.5,              # 水平翻转
        
        # 小目标增强策略
        'mosaic': 0.8,              # 马赛克增强
        'mixup': 0.1,               # 轻微混合，保持缺陷特征
        'copy_paste': 0.2,          # 复制粘贴增强微小缺陷
        
        # 损失函数权重 - 针对小目标优化
        'box': 7.5,                 # 提高边界框损失权重
        'cls': 0.7,                 # 分类损失权重
        'dfl': 1.5,                 # DFL损失权重
        
        # 训练设置
        'workers': 8,               # 数据加载线程
        'patience': 100,            # 早停耐心值
        'save_period': 10,          # 保存周期
    }
    
    # 微缺陷优化模式
    if args.micro_optimize:
        print("🔍 Micro defect optimization enabled")
        hyp.update({
            'lr0': 0.0005,          # 更低学习率
            'box': 10.0,            # 更高边界框权重
            'copy_paste': 0.3,      # 更多复制粘贴
            'close_mosaic': 30,     # 保持mosaic更久
            'mixup': 0.05,          # 减少mixup，保护微小特征
        })

    print(f"\n📊 Training Configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch}")
    print(f"  - Image Size: {args.imgsz}")
    print(f"  - Learning Rate: {hyp['lr0']}")
    print(f"  - Optimizer: {hyp['optimizer']}")
    print(f"  - Micro Optimize: {args.micro_optimize}")
    print(f"  - Box Loss Weight: {hyp['box']} (higher for small targets)")
    print(f"  - Copy-Paste Prob: {hyp['copy_paste']} (micro defect augmentation)")

    # 开始训练
    if args.resume:
        print("🔄 Resuming training...")
        model.train(resume=True)
    else:
        print("🎯 Starting industrial training...")
        print("🔬 Features enabled:")
        print("   ✅ P1/P2双层微缺陷检测头")
        print("   ✅ 蛇形可变形卷积增强")
        print("   ✅ BiFPN特征融合")
        print("   ✅ 小目标优化损失函数")
        
        model.train(
            data=args.data,
            project=args.project,
            name=args.name,
            pretrained=False,  # 使用自定义权重
            **hyp
        )

    print("🎉 Training completed!")
    print(f"📁 Results saved to: {args.project}/{args.name}")


if __name__ == '__main__':
    main()