#!/usr/bin/env python3
"""
Quick test script - 快速测试脚本
用于验证环境设置和模块注册
"""

def quick_test():
    """快速测试"""
    print("🧪 Quick Test - X-ray Defect Detection Setup")
    print("="*50)
    
    # 测试1: Python环境
    print("1. Python环境检查...")
    import sys
    print(f"   Python版本: {sys.version}")
    
    # 测试2: 必要包检查
    print("\n2. 必要包检查...")
    required_packages = ['torch', 'torchvision', 'ultralytics', 'numpy', 'opencv-python']
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - 未安装")
            return False
    
    # 测试3: 自定义模块注册
    print("\n3. 自定义模块注册...")
    try:
        from register_modules import register_all_modules
        register_all_modules()
        print("   ✅ 模块注册完成")
    except Exception as e:
        print(f"   ❌ 模块注册失败: {e}")
        return False
    
    # 测试4: YOLO导入
    print("\n4. YOLO导入测试...")
    try:
        from ultralytics import YOLO
        print("   ✅ YOLO导入成功")
    except Exception as e:
        print(f"   ❌ YOLO导入失败: {e}")
        return False
    
    # 测试5: 配置文件检查
    print("\n5. 配置文件检查...")
    import os
    
    config_files = [
        'models/yolo11_snake_bifpn.yaml',
        'data/xray_defects.yaml'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"   ✅ {config_file}")
        else:
            print(f"   ❌ {config_file} - 文件不存在")
            return False
    
    # 测试6: 模型创建测试
    print("\n6. 模型创建测试...")
    try:
        model = YOLO('models/yolo11_snake_bifpn.yaml')
        print("   ✅ 模型创建成功")
    except KeyError as e:
        print(f"   ❌ 模块未注册: {e}")
        print("   💡 请运行: python test_registration.py")
        return False
    except Exception as e:
        print(f"   ❌ 模型创建失败: {e}")
        return False
    
    print("\n🎉 所有测试通过！环境设置正确。")
    print("\n🚀 下一步:")
    print("   1. 准备数据集到 data/xray_weld_defects/ 目录")
    print("   2. 运行训练: python train_xray_defect.py")
    
    return True


if __name__ == "__main__":
    import sys
    success = quick_test()
    
    if not success:
        print("\n❌ 测试失败，请检查上述错误并修复")
        print("\n🔧 常见解决方案:")
        print("   pip install -r requirements.txt")
        print("   python test_registration.py")
        sys.exit(1)
    else:
        print("\n✅ 测试成功，可以开始训练！")
        sys.exit(0)