#!/usr/bin/env python3
"""
Check Built-in Modules - 检查内置模块
查看ultralytics框架中已有的模块，避免重复定义
"""

def check_builtin_modules():
    """检查内置模块"""
    print("🔍 Checking Built-in Modules")
    print("="*40)
    
    try:
        # 导入ultralytics模块
        from ultralytics import YOLO
        import ultralytics.nn.modules as modules
        import ultralytics.nn.tasks as tasks
        
        print("✅ Ultralytics imported successfully")
        
        # 检查常用模块是否已存在
        common_modules = [
            'C3k2', 'C2f', 'Conv', 'SPPF', 'Detect', 'Concat',
            'SEAttention', 'CBAM', 'ECA', 'C2PSA'
        ]
        
        print("\n📋 Built-in modules check:")
        builtin_modules = []
        missing_modules = []
        
        for module_name in common_modules:
            if hasattr(modules, module_name):
                builtin_modules.append(module_name)
                print(f"   ✅ {module_name} - Built-in")
            elif hasattr(tasks, module_name):
                builtin_modules.append(module_name)
                print(f"   ✅ {module_name} - In tasks")
            else:
                missing_modules.append(module_name)
                print(f"   ❌ {module_name} - Missing")
        
        print(f"\n📊 Summary:")
        print(f"   Built-in: {len(builtin_modules)}")
        print(f"   Missing: {len(missing_modules)}")
        
        # 如果C3k2存在，我们就不需要自定义的
        if 'C3k2' in builtin_modules:
            print("\n💡 C3k2 is built-in, no need for custom implementation")
        else:
            print("\n⚠️ C3k2 not found, will use custom implementation")
        
        return builtin_modules, missing_modules
        
    except Exception as e:
        print(f"❌ Failed to check built-in modules: {e}")
        return [], []


def create_safe_config():
    """创建安全的配置文件，只使用确认可用的模块"""
    print("\n🛡️ Creating Safe Config")
    print("="*30)
    
    builtin_modules, missing_modules = check_builtin_modules()
    
    # 创建一个只使用内置模块的安全配置
    safe_config = """
# Safe YOLO11 config - only built-in modules
# 安全的YOLO11配置 - 只使用内置模块

nc: 5
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]     # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]    # 1-P2/4
  - [-1, 2, C2f, [128, 256]]      # 2 - Use built-in C2f
  - [-1, 1, Conv, [256, 3, 2]]    # 3-P3/8
  - [-1, 2, C2f, [256, 512]]      # 4
  - [-1, 1, Conv, [512, 3, 2]]    # 5-P4/16
  - [-1, 2, C2f, [512, 512]]      # 6
  - [-1, 1, Conv, [512, 3, 2]]    # 7-P5/32
  - [-1, 2, C2f, [512, 1024]]     # 8
  - [-1, 1, SPPF, [1024, 1024]]   # 9

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 10
  - [[-1, 6], 1, Concat, [1]]                   # 11
  - [-1, 2, C2f, [1536, 512]]                   # 12
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 13
  - [[-1, 4], 1, Concat, [1]]                   # 14
  - [-1, 2, C2f, [768, 256]]                    # 15
  
  - [-1, 1, Conv, [256, 3, 2]]                  # 16
  - [[-1, 12], 1, Concat, [1]]                  # 17
  - [-1, 2, C2f, [768, 512]]                    # 18
  
  - [-1, 1, Conv, [512, 3, 2]]                  # 19
  - [[-1, 9], 1, Concat, [1]]                   # 20
  - [-1, 2, C2f, [1536, 1024]]                  # 21

  - [[15, 18, 21], 1, Detect, [nc]]             # 22
"""
    
    # 保存安全配置
    with open('models/yolo11_safe.yaml', 'w') as f:
        f.write(safe_config)
    
    print("💾 Created safe config: models/yolo11_safe.yaml")
    
    # 测试安全配置
    try:
        from ultralytics import YOLO
        model = YOLO('models/yolo11_safe.yaml')
        print("✅ Safe config works!")
        return True
    except Exception as e:
        print(f"❌ Safe config failed: {e}")
        print("📋 Full error:")
        traceback.print_exc()
        return False


def step_by_step_debug():
    """逐步调试"""
    print("\n🔬 Step-by-Step Debug")
    print("="*30)
    
    # 1. 测试基础导入
    print("1. Testing basic imports...")
    try:
        import torch
        import torchvision
        print("   ✅ PyTorch")
    except Exception as e:
        print(f"   ❌ PyTorch: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("   ✅ Ultralytics")
    except Exception as e:
        print(f"   ❌ Ultralytics: {e}")
        return False
    
    # 2. 测试标准YOLO模型
    print("\n2. Testing standard YOLO model...")
    try:
        model = YOLO('yolo11n.yaml')  # 使用标准配置
        print("   ✅ Standard YOLO11 works")
    except Exception as e:
        print(f"   ❌ Standard YOLO11 failed: {e}")
        # 尝试从预训练模型
        try:
            model = YOLO('yolo11n.pt')
            print("   ✅ Pretrained YOLO11 works")
        except Exception as e2:
            print(f"   ❌ Pretrained YOLO11 also failed: {e2}")
            return False
    
    # 3. 测试自定义模块导入
    print("\n3. Testing custom module imports...")
    try:
        from snake_bifpn_modules import SnakeDeformableConv
        print("   ✅ Snake modules import")
    except Exception as e:
        print(f"   ❌ Snake modules: {e}")
        return False
    
    try:
        from advanced_modules import SEAttention
        print("   ✅ Advanced modules import")
    except Exception as e:
        print(f"   ❌ Advanced modules: {e}")
        return False
    
    # 4. 测试模块实例化
    print("\n4. Testing module instantiation...")
    try:
        import torch
        x = torch.randn(1, 128, 32, 32)
        
        # 测试蛇形卷积
        snake_conv = SnakeDeformableConv(128, 256)
        output = snake_conv(x)
        print(f"   ✅ SnakeDeformableConv: {x.shape} -> {output.shape}")
        
        # 测试SE注意力
        se_att = SEAttention()
        output = se_att(x)
        print(f"   ✅ SEAttention: {x.shape} -> {output.shape}")
        
    except Exception as e:
        print(f"   ❌ Module instantiation failed: {e}")
        traceback.print_exc()
        return False
    
    print("\n✅ All step-by-step tests passed!")
    return True


def main():
    """主函数"""
    print("🐛 Comprehensive Model Debug")
    print("="*60)
    
    # 1. 逐步调试
    step_success = step_by_step_debug()
    
    if not step_success:
        print("\n❌ Basic setup failed. Please fix the environment first.")
        return False
    
    # 2. 详细模型调试
    debug_success = debug_model_creation()
    
    # 3. 如果失败，测试安全配置
    if not debug_success:
        print("\n🛡️ Trying safe config as fallback...")
        safe_success = create_safe_config()
        
        if safe_success:
            print("\n💡 Safe config works! Issue is with custom modules in complex config.")
        else:
            print("\n❌ Even safe config fails. Environment issue.")
    
    print("\n" + "="*60)
    print("🔍 Debug Summary:")
    print(f"   Step-by-step test: {'✅' if step_success else '❌'}")
    print(f"   Complex config test: {'✅' if debug_success else '❌'}")
    
    if debug_success:
        print("\n🎉 All tests passed! Your setup is working correctly.")
    else:
        print("\n🔧 Issues found. Check the detailed error messages above.")
    
    return debug_success


if __name__ == "__main__":
    main()