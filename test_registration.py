#!/usr/bin/env python3
"""
Test script to verify custom module registration
测试脚本 - 验证自定义模块注册是否成功
"""

def test_module_registration():
    """测试模块注册"""
    print("🧪 Testing Custom Module Registration")
    print("="*50)
    
    # 1. 注册模块
    print("📦 Step 1: Registering modules...")
    try:
        from register_modules import register_all_modules
        success = register_all_modules()
        if not success:
            print("❌ Module registration failed!")
            return False
    except Exception as e:
        print(f"❌ Failed to import register_modules: {e}")
        return False
    
    # 2. 检查模块是否可访问
    print("\n🔍 Step 2: Checking module availability...")
    try:
        import ultralytics.nn.tasks as tasks
        
        # 定义要检查的模块
        modules_to_check = [
            'C3k2_SnakeDeformable',
            'BiFPNBlock', 
            'TripleBiFPN',
            'MicroDefectAttention',
            'EnhancedDetectHead',
            'SEAttention',
            'CBAM',
            'ECA'
        ]
        
        available_modules = []
        missing_modules = []
        
        for module_name in modules_to_check:
            if hasattr(tasks, module_name):
                available_modules.append(module_name)
                print(f"  ✅ {module_name}")
            else:
                missing_modules.append(module_name)
                print(f"  ❌ {module_name}")
        
        print(f"\n📊 Results: {len(available_modules)}/{len(modules_to_check)} modules available")
        
        if missing_modules:
            print(f"⚠️ Missing modules: {missing_modules}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to check modules: {e}")
        return False
    
    # 3. 测试YOLO导入
    print("\n🚀 Step 3: Testing YOLO import...")
    try:
        from ultralytics import YOLO
        print("✅ YOLO imported successfully")
    except Exception as e:
        print(f"❌ Failed to import YOLO: {e}")
        return False
    
    # 4. 测试模型配置文件
    print("\n📄 Step 4: Testing model config...")
    import os
    config_file = "models/yolo11_snake_bifpn.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return False
    else:
        print(f"✅ Config file found: {config_file}")
    
    # 5. 尝试创建模型（不训练）
    print("\n🏗️ Step 5: Testing model creation...")
    try:
        model = YOLO(config_file)
        print("✅ Model created successfully!")
        
        # 获取模型信息
        print(f"📊 Model info: {len(list(model.model.parameters()))} parameters")
        
    except KeyError as e:
        print(f"❌ KeyError when creating model: {e}")
        print("💡 This suggests a module is not properly registered")
        return False
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return False
    
    print("\n🎉 All tests passed! Custom modules are properly registered.")
    return True


def main():
    """主函数"""
    success = test_module_registration()
    
    if success:
        print("\n✅ Registration test PASSED")
        print("🚀 You can now run: python train_xray_defect.py")
    else:
        print("\n❌ Registration test FAILED")
        print("🔧 Please check the error messages above and fix the issues")
    
    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)