#!/usr/bin/env python3
"""
Debug Model Creation - 调试模型创建
详细捕获和分析模型创建错误
"""

import traceback
import sys
import os


def debug_model_creation():
    """详细调试模型创建过程"""
    print("🔍 Debug Model Creation")
    print("="*50)
    
    try:
        # 1. 注册模块
        print("Step 1: Registering modules...")
        from register_modules import register_all_modules
        register_all_modules()
        
        # 2. 导入YOLO
        print("\nStep 2: Importing YOLO...")
        from ultralytics import YOLO
        
        # 3. 检查配置文件
        config_file = 'models/yolo11_snake_bifpn_simple.yaml'
        print(f"\nStep 3: Checking config file: {config_file}")
        
        if not os.path.exists(config_file):
            print(f"❌ File not found: {config_file}")
            return False
            
        # 4. 检查YAML语法
        print("Step 4: Validating YAML syntax...")
        import yaml
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print("✅ YAML syntax is valid")
            print(f"   NC: {config.get('nc', 'Not found')}")
            print(f"   Backbone layers: {len(config.get('backbone', []))}")
            print(f"   Head layers: {len(config.get('head', []))}")
        except yaml.YAMLError as e:
            print(f"❌ YAML syntax error: {e}")
            return False
        except Exception as e:
            print(f"❌ YAML loading error: {e}")
            return False
        
        # 5. 检查模块可用性
        print("\nStep 5: Checking module availability...")
        import ultralytics.nn.tasks as tasks
        
        # 从YAML中提取使用的模块
        used_modules = set()
        for layer in config.get('backbone', []):
            if len(layer) >= 3:
                module_name = layer[2]
                used_modules.add(module_name)
        for layer in config.get('head', []):
            if len(layer) >= 3:
                module_name = layer[2]
                used_modules.add(module_name)
        
        print(f"   Modules used in YAML: {sorted(used_modules)}")
        
        missing_modules = []
        for module_name in used_modules:
            if module_name.startswith('nn.'):
                continue  # Skip PyTorch modules
            if hasattr(tasks, module_name):
                print(f"   ✅ {module_name}")
            else:
                print(f"   ❌ {module_name}")
                missing_modules.append(module_name)
        
        if missing_modules:
            print(f"⚠️ Missing modules: {missing_modules}")
            return False
        
        # 6. 尝试创建模型
        print("\nStep 6: Creating model...")
        try:
            model = YOLO(config_file)
            print("✅ Model created successfully!")
            
            # 获取详细信息
            total_params = sum(p.numel() for p in model.model.parameters())
            print(f"📊 Total parameters: {total_params:,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model creation failed with detailed error:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            
            # 打印完整的traceback
            print("\n📋 Full traceback:")
            traceback.print_exc()
            
            return False
            
    except Exception as e:
        print(f"❌ Debug process failed: {e}")
        print("\n📋 Full traceback:")
        traceback.print_exc()
        return False


def test_minimal_config():
    """测试最小化配置"""
    print("\n" + "="*50)
    print("Testing Minimal Config")
    print("="*50)
    
    # 创建一个最小化的YAML配置
    minimal_config = """
# Minimal YOLO11 config for testing
nc: 5
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]     # 0
  - [-1, 1, Conv, [128, 3, 2]]    # 1
  - [-1, 1, SEAttention, []]      # 2
  - [-1, 2, C3k2, [128, 256]]     # 3
  - [-1, 1, Conv, [256, 3, 2]]    # 4
  - [-1, 2, C3k2, [256, 512]]     # 5
  - [-1, 1, Conv, [512, 3, 2]]    # 6
  - [-1, 2, C3k2, [512, 1024]]    # 7
  - [-1, 1, SPPF, [1024, 1024]]   # 8

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 9
  - [[-1, 6], 1, Concat, [1]]                   # 10
  - [-1, 2, C3k2, [1536, 512]]                  # 11
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 12
  - [[-1, 4], 1, Concat, [1]]                   # 13
  - [-1, 2, C3k2, [768, 256]]                   # 14
  
  - [-1, 1, Conv, [256, 3, 2]]                  # 15
  - [[-1, 11], 1, Concat, [1]]                  # 16
  - [-1, 2, C3k2, [768, 512]]                   # 17
  
  - [-1, 1, Conv, [512, 3, 2]]                  # 18
  - [[-1, 8], 1, Concat, [1]]                   # 19
  - [-1, 2, C3k2, [1536, 1024]]                 # 20

  - [[14, 17, 20], 1, Detect, [nc]]             # 21
"""
    
    # 保存最小化配置
    with open('models/yolo11_minimal.yaml', 'w') as f:
        f.write(minimal_config)
    
    print("💾 Created minimal config: models/yolo11_minimal.yaml")
    
    # 测试最小化配置
    try:
        from ultralytics import YOLO
        model = YOLO('models/yolo11_minimal.yaml')
        print("✅ Minimal config works!")
        return True
    except Exception as e:
        print(f"❌ Minimal config failed: {e}")
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🐛 Model Creation Debug Suite")
    print("="*60)
    
    # 1. 详细调试
    debug_success = debug_model_creation()
    
    # 2. 如果主配置失败，测试最小化配置
    if not debug_success:
        print("\n🔬 Trying minimal config as fallback...")
        minimal_success = test_minimal_config()
        
        if minimal_success:
            print("\n💡 Minimal config works! The issue is with the complex config.")
            print("🔧 Recommendation: Start with minimal config and gradually add modules")
        else:
            print("\n❌ Even minimal config fails. Check basic setup.")
    
    print("\n" + "="*60)
    if debug_success:
        print("🎉 Debug completed - Model creation works!")
    else:
        print("❌ Debug completed - Issues found")
        print("\n📋 Next steps:")
        print("1. Check the detailed error messages above")
        print("2. Fix any missing modules or syntax errors")
        print("3. Try the minimal config first")
        print("4. Gradually add complex modules")
    
    return debug_success


if __name__ == "__main__":
    main()