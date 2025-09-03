import os
import argparse

# Import modules
print("📦 Loading modules...")
import core_modules_final
import advanced_modules

from ultralytics import YOLO
print("✅ YOLO imported")


def parse_args():
    parser = argparse.ArgumentParser(description='X-ray Defect Detection - Fixed Scales')
    parser.add_argument('--model', type=str, default='models/yolo11_no_scales.yaml')
    parser.add_argument('--data', type=str, default='data/xray_defects.yaml')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='xray-fixed-scales')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--micro-optimize', action='store_true')
    parser.add_argument('--scale', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'], 
                       help='Model scale (explicitly specify to avoid auto-detection)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("🚀 X-ray Defect Detection - Fixed Scales Training")
    print("🔬 Four Core Functions with Corrected Modules:")
    print("   1. ✅ MicroDefectHead - 专用微缺陷检测头")
    print("   2. ✅ SnakeConv - 蛇形可变形卷积")
    print("   3. ✅ BiFPNSimple - 双向特征金字塔")
    print("   4. ✅ EnhancedC2f - 增强C2f模块")
    print(f"Model: {args.model}")
    print(f"Scale: {args.scale} (explicitly specified)")
    print(f"Data: {args.data}")
    
    # 验证配置文件
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("🔧 Available configs:")
        for f in os.listdir('models'):
            if f.endswith('.yaml'):
                print(f"   - models/{f}")
        return
    
    # 创建模型 - 明确指定scale避免警告
    print(f"🏗️ Creating model with scale '{args.scale}'...")
    try:
        # 方法1: 先从YAML创建，然后指定scale
        model = YOLO(args.model)
        
        # 手动设置scale避免自动检测
        if hasattr(model.model, 'args'):
            model.model.args.scale = args.scale
            
        print("✅ Model created successfully!")
        
        # 显示详细信息
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"📊 Parameters: {total_params:,}")
        print(f"📊 Size: {total_params * 4 / 1024 / 1024:.1f} MB")
        print(f"📊 Scale: {args.scale}")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        
        # 尝试fallback配置
        print("\n🚨 Trying fallback configuration...")
        fallback_configs = [
            'models/yolo11_fallback.yaml',
            'yolo11n.yaml',  # 标准配置
        ]
        
        for fallback in fallback_configs:
            try:
                print(f"   Trying {fallback}...")
                model = YOLO(fallback)
                print(f"   ✅ {fallback} works!")
                args.model = fallback  # 更新模型路径
                break
            except Exception as e2:
                print(f"   ❌ {fallback} failed: {str(e2)[:50]}...")
                continue
        else:
            print("❌ All fallback configs failed!")
            return
    
    # 加载权重
    if args.weights and os.path.exists(args.weights):
        model.load(args.weights)
        print(f"✅ Loaded weights: {args.weights}")

    # 优化的训练超参数
    hyp = {
        'imgsz': args.imgsz,
        'batch': args.batch,
        'epochs': args.epochs,
        'device': args.device,
        'optimizer': 'AdamW',
        'cos_lr': True,
        'close_mosaic': 20,
        
        # 针对小目标的学习率
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # X光图像增强
        'hsv_h': 0.005,
        'hsv_s': 0.3,
        'hsv_v': 0.5,
        'degrees': 3.0,
        'translate': 0.05,
        'scale': 0.4,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 0.8,
        'mixup': 0.1,
        'copy_paste': 0.2,
        
        # 小目标损失优化
        'box': 7.5,
        'cls': 0.7,
        'dfl': 1.5,
        
        'workers': 8,
        'patience': 100,
        'save_period': 10,
    }
    
    # 微缺陷优化
    if args.micro_optimize:
        print("🔍 Micro defect optimization enabled")
        hyp.update({
            'lr0': 0.0005,
            'box': 10.0,
            'copy_paste': 0.3,
            'close_mosaic': 30,
        })

    print(f"\n📊 Training Configuration:")
    print(f"  - Model: {args.model}")
    print(f"  - Scale: {args.scale}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch: {args.batch}")
    print(f"  - Image Size: {args.imgsz}")
    print(f"  - Learning Rate: {hyp['lr0']}")
    print(f"  - Box Weight: {hyp['box']} (small object optimization)")

    # 开始训练
    if args.resume:
        print("🔄 Resuming training...")
        model.train(resume=True)
    else:
        print("🎯 Starting training with fixed scales...")
        
        model.train(
            data=args.data,
            project=args.project,
            name=args.name,
            pretrained=False,
            **hyp
        )

    print("🎉 Training completed!")


if __name__ == '__main__':
    main()