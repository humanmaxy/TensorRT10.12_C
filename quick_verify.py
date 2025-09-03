#!/usr/bin/env python3
"""
Quick Verify - 快速验证简化版配置
专门验证简化版YAML配置是否能正常工作
"""

def quick_verify():
    """快速验证"""
    print("⚡ Quick Verify - Simple Config Test")
    print("="*40)
    
    # 1. 注册模块
    print("📦 Registering modules...")
    try:
        from register_modules import register_all_modules
        register_all_modules()
    except Exception as e:
        print(f"❌ Module registration failed: {e}")
        return False
    
    # 2. 导入YOLO
    print("\n🚀 Importing YOLO...")
    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"❌ YOLO import failed: {e}")
        return False
    
    # 3. 测试简化版配置
    config_file = 'models/yolo11_snake_bifpn_simple.yaml'
    print(f"\n🏗️ Testing {config_file}...")
    
    import os
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return False
    
    try:
        model = YOLO(config_file)
        print("✅ Model created successfully!")
        
        # 获取基本信息
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"📊 Parameters: {total_params:,}")
        print(f"📊 Size: {total_params * 4 / 1024 / 1024:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        
        # 简化的错误分析
        error_str = str(e).lower()
        if "cbam" in error_str and len(error_str) > 100:
            print("💡 This is the string repetition error")
            print("🔧 Solution: The simple config should fix this")
        elif "missing" in error_str and "argument" in error_str:
            print("💡 This is a parameter mismatch error")
            print("🔧 Solution: Check module parameter format")
        elif "keyerror" in error_str:
            print("💡 This is a module registration error")
            print("🔧 Solution: Check if all modules are registered")
        
        return False


def main():
    """主函数"""
    success = quick_verify()
    
    print("\n" + "="*40)
    if success:
        print("🎉 Quick verify PASSED!")
        print("✅ Simple config works correctly")
        print("\n🚀 Ready to train:")
        print("   python train_xray_defect.py")
    else:
        print("❌ Quick verify FAILED!")
        print("🔧 Try running: python test_model_creation.py")
        print("   for detailed error analysis")
    
    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)