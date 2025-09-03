import os
import argparse

# Import core modules BEFORE YOLO
print("📦 Loading core modules...")
import core_modules_final    # 四大功能核心模块
import advanced_modules      # 现有增强模块

from ultralytics import YOLO
print("✅ YOLO imported successfully")


def parse_args():
    parser = argparse.ArgumentParser(description='Industrial X-ray Defect Detection - Core Final')
    parser.add_argument('--model', type=str, default='models/yolo11_core_final.yaml')
    parser.add_argument('--data', type=str, default='data/xray_defects.yaml')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='xray-core-final')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--micro-optimize', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("🚀 Industrial X-ray Defect Detection - Core Final")
    print("🔬 Four Core Functions Implementation:")
    print("   1. ✅ 专用微缺陷检测头 (P1/P2双层，15微米级别)")
    print("   2. ✅ 蛇形卷积 (适应不规则缺陷)")
    print("   3. ✅ 简化BiFPN (多尺度特征融合)")
    print("   4. ✅ 小目标优化训练 (损失函数+数据增强)")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    
    # 验证模型创建
    print("🏗️ Creating model...")
    try:
        model = YOLO(args.model)
        print("✅ Model created successfully!")
        
        # 显示模型信息
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"📊 Parameters: {total_params:,}")
        print(f"📊 Size: {total_params * 4 / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        print("🔧 Please check the configuration file")
        return
    
    # 加载权重
    if args.weights and os.path.exists(args.weights):
        model.load(args.weights)
        print(f"✅ Loaded weights: {args.weights}")

    # 训练超参数 - 针对微缺陷优化（功能4）
    hyp = {
        'imgsz': args.imgsz,
        'batch': args.batch,
        'epochs': args.epochs,
        'device': args.device,
        'optimizer': 'AdamW',
        'cos_lr': True,
        'close_mosaic': 20,
        
        # 微缺陷优化学习率
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        
        # X光图像数据增强
        'hsv_h': 0.005,      # 最小色调变化
        'hsv_s': 0.3,        # 适度饱和度
        'hsv_v': 0.5,        # 亮度变化（模拟曝光）
        'degrees': 3.0,      # 小角度旋转
        'translate': 0.05,   # 小幅平移
        'scale': 0.4,        # 缩放增强
        'shear': 0.0,        # 不使用剪切
        'perspective': 0.0,  # 不使用透视
        'flipud': 0.5,       # 垂直翻转
        'fliplr': 0.5,       # 水平翻转
        'mosaic': 0.8,       # 马赛克增强
        'mixup': 0.1,        # 轻微混合
        'copy_paste': 0.2,   # 复制粘贴增强微缺陷
        
        # 小目标损失权重优化（功能4）
        'box': 7.5,          # 边界框损失权重
        'cls': 0.7,          # 分类损失权重
        'dfl': 1.5,          # DFL损失权重
        
        'workers': 8,
        'patience': 100,
        'save_period': 10,
    }
    
    # 微缺陷优化模式
    if args.micro_optimize:
        print("🔍 Micro defect optimization enabled")
        hyp.update({
            'lr0': 0.0005,       # 更低学习率
            'box': 10.0,         # 更高边界框权重
            'copy_paste': 0.3,   # 更多微缺陷增强
            'close_mosaic': 30,  # 保持mosaic更久
            'mixup': 0.05,       # 减少mixup保护微特征
        })

    print(f"\n📊 Training Configuration (Four Functions Optimized):")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch: {args.batch}")
    print(f"  - Image Size: {args.imgsz} (P1层可达{args.imgsz*2})")
    print(f"  - Learning Rate: {hyp['lr0']} (微缺陷优化)")
    print(f"  - Box Loss Weight: {hyp['box']} (小目标优化)")
    print(f"  - Copy-Paste: {hyp['copy_paste']} (微缺陷增强)")
    print(f"  - Optimizer: {hyp['optimizer']} (小目标友好)")

    # 开始训练
    if args.resume:
        print("🔄 Resuming training...")
        model.train(resume=True)
    else:
        print("🎯 Starting industrial training with four core functions...")
        
        model.train(
            data=args.data,
            project=args.project,
            name=args.name,
            pretrained=False,
            **hyp
        )

    print("🎉 Training completed!")
    print(f"📁 Results: {args.project}/{args.name}")
    print("🔬 Features trained:")
    print("   ✅ P1层超高分辨率检测 (15微米级别)")
    print("   ✅ 蛇形卷积适应不规则缺陷")
    print("   ✅ BiFPN多尺度特征融合")
    print("   ✅ 小目标优化损失和增强")


if __name__ == '__main__':
    main()