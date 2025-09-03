#!/usr/bin/env python3
"""
Test Scales - 测试不同的scales配置
找到正确的scales避免通道数错误
"""

import os


def test_scales_config():
    """测试不同的scales配置"""
    print("🧪 Testing Different Scales Configurations")
    print("="*50)
    
    # 1. 注册模块
    print("📦 Registering modules...")
    try:
        import core_modules_final
        import advanced_modules
        print("✅ Modules registered")
    except Exception as e:
        print(f"❌ Module registration failed: {e}")
        return False
    
    # 2. 导入YOLO
    try:
        from ultralytics import YOLO
        print("✅ YOLO imported")
    except Exception as e:
        print(f"❌ YOLO import failed: {e}")
        return False
    
    # 3. 测试不同配置
    configs_to_test = [
        ('models/yolo11_no_scales.yaml', 'No Scales'),
        ('models/yolo11_corrected_scales.yaml', 'Corrected Scales'),
        ('models/yolo11_standard_scales.yaml', 'Standard Scales'),
        ('models/yolo11_ultra_simple.yaml', 'Ultra Simple'),
    ]
    
    working_configs = []
    
    for config_file, description in configs_to_test:
        print(f"\n🧪 Testing {description}...")
        
        if not os.path.exists(config_file):
            print(f"   ⚠️ File not found: {config_file}")
            continue
        
        try:
            # 尝试创建模型
            model = YOLO(config_file)
            print(f"   ✅ {description} works!")
            
            # 获取参数信息
            total_params = sum(p.numel() for p in model.model.parameters())
            print(f"   📊 Parameters: {total_params:,}")
            
            working_configs.append((config_file, description))
            
        except Exception as e:
            print(f"   ❌ {description} failed: {str(e)[:100]}...")
            continue
    
    # 4. 报告结果
    print(f"\n📊 Test Results:")
    print(f"   Working configs: {len(working_configs)}")
    
    if working_configs:
        best_config = working_configs[0]
        print(f"\n🎉 Recommended config: {best_config[0]}")
        print(f"   Description: {best_config[1]}")
        
        # 更新训练脚本
        update_training_script(best_config[0])
        
        return True
    else:
        print(f"\n❌ No working configurations found!")
        create_minimal_fallback()
        return False


def update_training_script(config_file):
    """更新训练脚本使用最佳配置"""
    print(f"\n🔄 Updating training script...")
    
    training_scripts = [
        'train_ultra_simple.py',
        'train_core_final.py'
    ]
    
    for script in training_scripts:
        if os.path.exists(script):
            try:
                with open(script, 'r') as f:
                    content = f.read()
                
                # 查找并替换默认配置
                import re
                pattern = r"default='models/[^']*\.yaml'"
                replacement = f"default='{config_file}'"
                
                new_content = re.sub(pattern, replacement, content)
                
                with open(script, 'w') as f:
                    f.write(new_content)
                
                print(f"✅ Updated {script}")
                
            except Exception as e:
                print(f"⚠️ Failed to update {script}: {e}")


def create_minimal_fallback():
    """创建最小fallback配置"""
    print("\n🚨 Creating Minimal Fallback")
    print("="*30)
    
    # 最基础的YOLO11配置，不使用任何自定义模块
    fallback_config = """
# Minimal Fallback - 最小回退配置
# 只使用内置模块，确保100%工作

nc: 5

backbone:
  - [-1, 1, Conv, [64, 3, 2]]     # 0
  - [-1, 1, Conv, [128, 3, 2]]    # 1
  - [-1, 2, C2f, [128, 256]]      # 2
  - [-1, 1, Conv, [256, 3, 2]]    # 3
  - [-1, 2, C2f, [256, 512]]      # 4
  - [-1, 1, Conv, [512, 3, 2]]    # 5
  - [-1, 2, C2f, [512, 512]]      # 6
  - [-1, 1, Conv, [512, 3, 2]]    # 7
  - [-1, 2, C2f, [512, 1024]]     # 8
  - [-1, 1, SPPF, [1024, 1024]]   # 9

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 10
  - [[-1, 6], 1, Concat, [1]]                   # 11
  - [-1, 2, C2f, [1536, 512]]                   # 12
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 13
  - [[-1, 4], 1, Concat, [1]]                   # 14
  - [-1, 2, C2f, [1024, 256]]                   # 15
  
  # P2层 - 小目标检测
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 16
  - [[-1, 2], 1, Concat, [1]]                   # 17
  - [-1, 2, C2f, [512, 128]]                    # 18
  
  - [-1, 1, Conv, [128, 3, 2]]                  # 19
  - [[-1, 15], 1, Concat, [1]]                  # 20
  - [-1, 2, C2f, [384, 256]]                    # 21
  
  - [-1, 1, Conv, [256, 3, 2]]                  # 22
  - [[-1, 12], 1, Concat, [1]]                  # 23
  - [-1, 2, C2f, [768, 512]]                    # 24
  
  - [-1, 1, Conv, [512, 3, 2]]                  # 25
  - [[-1, 9], 1, Concat, [1]]                   # 26
  - [-1, 2, C2f, [1536, 1024]]                  # 27

  # 四尺度检测（包含P2小目标）
  - [[18, 21, 24, 27], 1, Detect, [nc]]         # 28
"""
    
    with open('models/yolo11_fallback.yaml', 'w') as f:
        f.write(fallback_config)
    
    print("💾 Created fallback config: models/yolo11_fallback.yaml")
    
    # 测试fallback
    try:
        from ultralytics import YOLO
        model = YOLO('models/yolo11_fallback.yaml')
        print("✅ Fallback config works!")
        return True
    except Exception as e:
        print(f"❌ Even fallback failed: {e}")
        return False


def main():
    """主函数"""
    print("🔧 Scales Configuration Test")
    print("="*60)
    
    success = test_scales_config()
    
    print("\n" + "="*60)
    if success:
        print("🎉 Found working scales configuration!")
        print("\n🚀 Ready to train:")
        print("   python train_ultra_simple.py --data data/xray_defects.yaml")
    else:
        print("❌ No working scales configuration found!")
        print("🔧 Try using the fallback config")
    
    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)