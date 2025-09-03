#!/usr/bin/env python3
"""
Test Model Creation - 测试模型创建
验证YAML配置文件和模块参数是否正确
"""

def test_model_creation():
    """测试模型创建"""
    print("🧪 Testing Model Creation")
    print("="*50)
    
    # 1. 注册模块
    print("📦 Step 1: Registering modules...")
    try:
        from register_modules import register_all_modules
        register_all_modules()
    except Exception as e:
        print(f"❌ Failed to register modules: {e}")
        return False
    
    # 2. 导入YOLO
    print("\n🚀 Step 2: Importing YOLO...")
    try:
        from ultralytics import YOLO
        print("✅ YOLO imported successfully")
    except Exception as e:
        print(f"❌ Failed to import YOLO: {e}")
        return False
    
    # 3. 测试配置文件
    config_files = [
        'models/yolo11_snake_bifpn_simple.yaml',  # 简化版 - 推荐
        'models/yolo11_snake_bifpn_fixed.yaml',   # 修复版
        'models/yolo11_snake_bifpn.yaml'          # 原版本
    ]
    
    for config_file in config_files:
        print(f"\n🏗️ Step 3: Testing {config_file}...")
        
        # 检查文件是否存在
        import os
        if not os.path.exists(config_file):
            print(f"❌ Config file not found: {config_file}")
            continue
        
        # 尝试创建模型
        try:
            print(f"   Creating model from {config_file}...")
            model = YOLO(config_file)
            print(f"   ✅ Model created successfully!")
            
            # 获取模型参数信息
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            
            print(f"   📊 Total parameters: {total_params:,}")
            print(f"   📊 Trainable parameters: {trainable_params:,}")
            print(f"   📊 Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to create model: {e}")
            print(f"   💡 Error type: {type(e).__name__}")
            
            # 提供具体的错误分析
            if "missing" in str(e) and "required positional argument" in str(e):
                print("   🔧 This suggests incorrect parameter format in YAML")
            elif "KeyError" in str(e):
                print("   🔧 This suggests a module is not properly registered")
            elif "channels" in str(e).lower():
                print("   🔧 This suggests channel dimension mismatch")
            
            continue
    
    return False


def test_individual_modules():
    """测试单个模块创建"""
    print("\n🔍 Testing Individual Modules")
    print("="*30)
    
    try:
        from snake_bifpn_modules import (
            SnakeDeformableConv, C3k2_SnakeDeformable,
            BiFPNBlock, TripleBiFPN, MicroDefectAttention
        )
        from simple_modules import C3k2
        
        import torch
        
        # 测试输入
        test_input = torch.randn(2, 128, 32, 32)
        
        # 测试各个模块
        modules_to_test = [
            ("C3k2", C3k2, [128, 256]),
            ("SnakeDeformableConv", SnakeDeformableConv, [128, 256]),
            ("C3k2_SnakeDeformable", C3k2_SnakeDeformable, [128, 256]),
            ("BiFPNBlock", BiFPNBlock, [128, 256]),
            ("TripleBiFPN", TripleBiFPN, [128, 256]),
            ("MicroDefectAttention", MicroDefectAttention, [128])
        ]
        
        for module_name, module_class, args in modules_to_test:
            try:
                module = module_class(*args)
                output = module(test_input)
                print(f"✅ {module_name}: {test_input.shape} -> {output.shape}")
            except Exception as e:
                print(f"❌ {module_name}: {e}")
                
    except Exception as e:
        print(f"❌ Failed to test individual modules: {e}")


def main():
    """主函数"""
    print("🧪 Model Creation Test Suite")
    print("="*60)
    
    # 测试模型创建
    model_success = test_model_creation()
    
    # 测试单个模块
    test_individual_modules()
    
    print("\n" + "="*60)
    if model_success:
        print("🎉 Model creation test PASSED!")
        print("✅ You can now run training with the fixed config:")
        print("   python train_xray_defect.py --model models/yolo11_snake_bifpn_fixed.yaml")
    else:
        print("❌ Model creation test FAILED!")
        print("🔧 Please check the error messages above")
    
    return model_success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)