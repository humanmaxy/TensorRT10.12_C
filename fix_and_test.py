#!/usr/bin/env python3
"""
Fix and Test - 修复并测试
一站式解决所有模型创建问题
"""

import os
import sys
import traceback


def fix_and_test():
    """修复并测试所有配置"""
    print("🔧 Fix and Test - One-Stop Solution")
    print("="*50)
    
    # 1. 注册模块
    print("📦 Step 1: Registering modules...")
    try:
        from register_modules import register_all_modules
        register_all_modules()
    except Exception as e:
        print(f"❌ Module registration failed: {e}")
        return False
    
    # 2. 导入YOLO
    print("\n🚀 Step 2: Importing YOLO...")
    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"❌ YOLO import failed: {e}")
        return False
    
    # 3. 测试配置文件（按安全性排序）
    configs_to_test = [
        ('models/yolo11_dimension_safe.yaml', 'Dimension Safe'),
        ('models/yolo11_progressive.yaml', 'Progressive'),
        ('models/yolo11_snake_bifpn_safe.yaml', 'Snake BiFPN Safe'),
        ('models/yolo11_minimal.yaml', 'Minimal'),
    ]
    
    working_configs = []
    
    for config_file, description in configs_to_test:
        print(f"\n🧪 Step 3.{len(working_configs)+1}: Testing {description}...")
        
        if not os.path.exists(config_file):
            print(f"   ⚠️ File not found: {config_file}")
            continue
        
        try:
            model = YOLO(config_file)
            print(f"   ✅ {description} config works!")
            
            # 获取模型信息
            total_params = sum(p.numel() for p in model.model.parameters())
            print(f"   📊 Parameters: {total_params:,}")
            
            working_configs.append((config_file, description))
            
        except Exception as e:
            print(f"   ❌ {description} failed: {str(e)[:100]}...")
            continue
    
    # 4. 报告结果
    print(f"\n📊 Test Results:")
    print(f"   Working configs: {len(working_configs)}")
    print(f"   Failed configs: {len(configs_to_test) - len(working_configs)}")
    
    if working_configs:
        print(f"\n✅ Recommended config: {working_configs[0][0]}")
        print(f"   Description: {working_configs[0][1]}")
        
        # 更新训练脚本默认配置
        update_training_script(working_configs[0][0])
        
        return True
    else:
        print(f"\n❌ No working configs found!")
        create_emergency_config()
        return False


def update_training_script(config_file):
    """更新训练脚本的默认配置"""
    print(f"\n🔄 Updating training script to use: {config_file}")
    
    try:
        # 读取训练脚本
        with open('train_xray_defect.py', 'r') as f:
            content = f.read()
        
        # 更新默认配置路径
        old_default = "default='models/yolo11_snake_bifpn_safe.yaml'"
        new_default = f"default='{config_file}'"
        
        if old_default in content:
            content = content.replace(old_default, new_default)
            
            with open('train_xray_defect.py', 'w') as f:
                f.write(content)
            
            print(f"✅ Updated train_xray_defect.py default config to: {config_file}")
        else:
            print("⚠️ Could not update training script automatically")
            
    except Exception as e:
        print(f"⚠️ Failed to update training script: {e}")


def create_emergency_config():
    """创建紧急配置 - 最基础的YOLO11"""
    print("\n🚨 Creating Emergency Config")
    print("="*30)
    
    emergency_config = """
# Emergency YOLO11 Config - Minimal working version
nc: 5

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C2f, [128, 256]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C2f, [256, 512]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C2f, [512, 1024]]
  - [-1, 1, SPPF, [1024, 1024]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 5], 1, Concat, [1]]
  - [-1, 2, C2f, [1536, 512]]
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 3], 1, Concat, [1]]
  - [-1, 2, C2f, [768, 256]]
  
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C2f, [768, 512]]
  
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 7], 1, Concat, [1]]
  - [-1, 2, C2f, [1536, 1024]]

  - [[13, 16, 19], 1, Detect, [nc]]
"""
    
    with open('models/yolo11_emergency.yaml', 'w') as f:
        f.write(emergency_config)
    
    print("💾 Created emergency config: models/yolo11_emergency.yaml")
    
    try:
        from ultralytics import YOLO
        model = YOLO('models/yolo11_emergency.yaml')
        print("✅ Emergency config works!")
        return True
    except Exception as e:
        print(f"❌ Emergency config failed: {e}")
        return False


def main():
    """主函数"""
    success = fix_and_test()
    
    print("\n" + "="*50)
    if success:
        print("🎉 Fix and Test COMPLETED!")
        print("\n🚀 Ready to train:")
        print("   python train_xray_defect.py")
        print("\n📋 Available configs (in order of preference):")
        print("   1. models/yolo11_dimension_safe.yaml")
        print("   2. models/yolo11_progressive.yaml") 
        print("   3. models/yolo11_snake_bifpn_safe.yaml")
        print("   4. models/yolo11_emergency.yaml")
    else:
        print("❌ Fix and Test FAILED!")
        print("\n🔧 Emergency actions:")
        print("   1. Check ultralytics version: pip show ultralytics")
        print("   2. Reinstall: pip uninstall ultralytics && pip install ultralytics")
        print("   3. Use emergency config: models/yolo11_emergency.yaml")
    
    return success


if __name__ == "__main__":
    main()