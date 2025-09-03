import os
import argparse

# IMPORTANT: Register custom modules BEFORE importing YOLO
print("📦 Registering custom modules...")

# Use the comprehensive registration system
from register_modules import register_all_modules
register_all_modules()

# Now import YOLO after modules are registered
from ultralytics import YOLO
print("✅ YOLO imported successfully")


def parse_args():
    parser = argparse.ArgumentParser(description='X-ray Weld Defect Detection with Snake Conv + BiFPN')
    parser.add_argument('--model', type=str, default='models/yolo11_snake_bifpn.yaml', help='Model config path')
    parser.add_argument('--data', type=str, default='data/xray_defects.yaml', help='Dataset config path')
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='Device (0, 1, cpu)')
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory')
    parser.add_argument('--name', type=str, default='xray-snake-bifpn', help='Experiment name')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--weights', type=str, default='', help='Pretrained weights path')
    parser.add_argument('--micro-optimize', action='store_true', help='Enable micro defect optimization')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("🚀 X-ray Weld Defect Detection Training")
    print("🐍 Snake Deformable Convolution + 🔄 BiFPN + 🔍 Micro Defect Detection")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    
    # Verify custom modules are available
    print("🔍 Verifying custom modules...")
    try:
        import ultralytics.nn.tasks as tasks
        key_modules = ['C3k2_SnakeDeformable', 'BiFPNBlock', 'MicroDefectAttention', 'SEAttention']
        for module_name in key_modules:
            if hasattr(tasks, module_name):
                print(f"  ✅ {module_name} - Available")
            else:
                print(f"  ❌ {module_name} - Missing")
                
        # Also check if model file exists
        if not os.path.exists(args.model):
            print(f"❌ Model file not found: {args.model}")
            return
        else:
            print(f"✅ Model file found: {args.model}")
            
    except Exception as e:
        print(f"⚠️ Module verification failed: {e}")
    
    # Create model from YAML config
    print("🏗️ Creating model...")
    try:
        model = YOLO(args.model)
        print("✅ Model created successfully!")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure all custom modules are properly registered")
        print("2. Check if the YAML file syntax is correct")
        print("3. Verify all module names in the YAML match registered modules")
        return
    
    # Load pretrained weights if provided
    if args.weights and os.path.exists(args.weights):
        model.load(args.weights)
        print(f"✅ Loaded pretrained weights: {args.weights}")

    # Training hyperparameters optimized for micro defects
    hyp = {
        'imgsz': args.imgsz,
        'batch': args.batch,
        'epochs': args.epochs,
        'device': args.device,
        'optimizer': 'AdamW',  # Better for small targets
        'lr0': 0.001,          # Lower learning rate for micro defects
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Data augmentation for X-ray images
        'hsv_h': 0.01,         # Minimal hue variation for X-ray
        'hsv_s': 0.3,          # Moderate saturation
        'hsv_v': 0.6,          # Higher value variation (exposure)
        'degrees': 5.0,        # Small rotation
        'translate': 0.05,     # Small translation
        'scale': 0.3,          # Scale augmentation
        'shear': 1.0,          # Minimal shear
        'perspective': 0.0,    # No perspective for X-ray
        'flipud': 0.5,         # Vertical flip
        'fliplr': 0.5,         # Horizontal flip
        'mosaic': 1.0,         # Mosaic augmentation
        'mixup': 0.05,         # Light mixup to preserve defect features
        'copy_paste': 0.1,     # Copy-paste for micro defects
        
        # Loss weights optimized for small defects
        'box': 7.5,            # Higher box loss weight
        'cls': 0.5,            # Standard classification weight
        'dfl': 1.5,            # DFL weight
        
        # Training settings
        'cos_lr': True,        # Cosine learning rate scheduler
        'close_mosaic': 20,    # Close mosaic in last 20 epochs
        'patience': 50,        # Early stopping patience
        'save_period': 10,     # Save every 10 epochs
        'workers': 8,          # Data loading workers
    }
    
    # Micro defect optimization
    if args.micro_optimize:
        print("🔍 Micro defect optimization enabled")
        hyp.update({
            'lr0': 0.0005,         # Even lower learning rate
            'box': 10.0,           # Higher box loss for small targets
            'close_mosaic': 30,    # Keep mosaic longer
            'copy_paste': 0.2,     # More copy-paste for micro defects
        })

    print(f"📊 Training Configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch}")
    print(f"  - Image Size: {args.imgsz}")
    print(f"  - Learning Rate: {hyp['lr0']}")
    print(f"  - Optimizer: {hyp['optimizer']}")
    print(f"  - Micro Optimize: {args.micro_optimize}")

    # Start training
    if args.resume:
        print("🔄 Resuming training...")
        model.train(resume=True)
    else:
        print("🎯 Starting training...")
        model.train(
            data=args.data,
            project=args.project,
            name=args.name,
            pretrained=False,  # Use custom weights only
            **hyp
        )

    print("🎉 Training completed!")


if __name__ == '__main__':
    main()