#!/usr/bin/env python3
"""
Debug Tensor Sizes - 调试tensor尺寸问题
专门解决 "Sizes of tensors must match" 错误
"""

import torch
import torch.nn as nn
import yaml


def analyze_tensor_flow():
    """分析tensor在网络中的流动"""
    print("🔍 Analyzing Tensor Flow")
    print("="*40)
    
    # 模拟网络前向传播，跟踪tensor尺寸变化
    print("📊 Simulating network forward pass...")
    
    # 输入tensor
    x = torch.randn(1, 3, 640, 640)
    print(f"Input: {x.shape}")
    
    # 模拟backbone
    print("\n🏗️ Backbone:")
    
    # Stem
    x = torch.randn(1, 64, 320, 320)   # Conv [64, 3, 2]
    print(f"After Conv1: {x.shape}")
    
    x = torch.randn(1, 128, 160, 160)  # Conv [128, 3, 2]
    print(f"After Conv2: {x.shape}")
    
    # Stage 1
    p2 = torch.randn(1, 256, 80, 80)   # After stage 1
    print(f"P2 (after stage 1): {p2.shape}")
    
    # Stage 2  
    p3 = torch.randn(1, 512, 40, 40)   # After stage 2
    print(f"P3 (after stage 2): {p3.shape}")
    
    # Stage 3
    p4 = torch.randn(1, 512, 20, 20)   # After stage 3
    print(f"P4 (after stage 3): {p4.shape}")
    
    # Stage 4
    p5 = torch.randn(1, 1024, 10, 10)  # After stage 4
    print(f"P5 (after stage 4): {p5.shape}")
    
    # 模拟head部分的concat操作
    print("\n🎯 Head (concat operations):")
    
    # P4 concat
    p5_up = torch.randn(1, 1024, 20, 20)  # Upsample P5 to P4 size
    p4_concat = torch.cat([p5_up, p4], dim=1)  # Should be [1, 1536, 20, 20]
    print(f"P4 concat: {p5_up.shape} + {p4.shape} = {p4_concat.shape}")
    
    # P3 concat
    p4_processed = torch.randn(1, 512, 40, 40)  # After processing P4
    p3_concat = torch.cat([p4_processed, p3], dim=1)  # Should be [1, 1024, 40, 40]
    print(f"P3 concat: {p4_processed.shape} + {p3.shape} = {p3_concat.shape}")
    
    # P2 concat - 这里可能有问题
    p3_processed = torch.randn(1, 256, 80, 80)  # After processing P3
    try:
        p2_concat = torch.cat([p3_processed, p2], dim=1)  # Should be [1, 512, 80, 80]
        print(f"P2 concat: {p3_processed.shape} + {p2.shape} = {p2_concat.shape}")
    except RuntimeError as e:
        print(f"❌ P2 concat failed: {e}")
        print(f"   P3_processed: {p3_processed.shape}")
        print(f"   P2: {p2.shape}")
        print("   💡 Size mismatch detected!")
    
    return True


def create_dimension_safe_config():
    """创建尺寸安全的配置"""
    print("\n🛡️ Creating Dimension-Safe Config")
    print("="*40)
    
    # 创建一个确保尺寸匹配的配置
    safe_config = """
# Dimension-Safe YOLO11 Config
# 尺寸安全的YOLO11配置，避免tensor尺寸不匹配

nc: 5
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]

# 简化的backbone，确保尺寸匹配
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]     # 0-P1/2  (320x320)
  - [-1, 1, Conv, [128, 3, 2]]    # 1-P2/4  (160x160)
  - [-1, 2, C2f, [128, 256]]      # 2       (160x160)
  - [-1, 1, Conv, [256, 3, 2]]    # 3-P3/8  (80x80)
  - [-1, 2, C2f, [256, 512]]      # 4       (80x80)
  - [-1, 1, Conv, [512, 3, 2]]    # 5-P4/16 (40x40)
  - [-1, 2, C2f, [512, 512]]      # 6       (40x40)
  - [-1, 1, Conv, [512, 3, 2]]    # 7-P5/32 (20x20)
  - [-1, 2, C2f, [512, 1024]]     # 8       (20x20)
  - [-1, 1, SPPF, [1024, 1024]]   # 9       (20x20)

# 标准的head结构，确保尺寸计算正确
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 10 -> (40x40)
  - [[-1, 6], 1, Concat, [1]]                   # 11 cat(1024+512=1536, 40, 40)
  - [-1, 2, C2f, [1536, 512]]                   # 12 (P4/16)
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 13 -> (80x80)
  - [[-1, 4], 1, Concat, [1]]                   # 14 cat(512+512=1024, 80, 80)
  - [-1, 2, C2f, [1024, 256]]                   # 15 (P3/8)
  
  - [-1, 1, Conv, [256, 3, 2]]                  # 16 -> (40x40)
  - [[-1, 12], 1, Concat, [1]]                  # 17 cat(256+512=768, 40, 40)
  - [-1, 2, C2f, [768, 512]]                    # 18 (P4/16)
  
  - [-1, 1, Conv, [512, 3, 2]]                  # 19 -> (20x20)
  - [[-1, 9], 1, Concat, [1]]                   # 20 cat(512+1024=1536, 20, 20)
  - [-1, 2, C2f, [1536, 1024]]                  # 21 (P5/32)

  # 三尺度检测头
  - [[15, 18, 21], 1, Detect, [nc]]             # 22 Detect(P3, P4, P5)
"""
    
    with open('models/yolo11_dimension_safe.yaml', 'w') as f:
        f.write(safe_config)
    
    print("💾 Created dimension-safe config: models/yolo11_dimension_safe.yaml")
    
    # 测试这个配置
    try:
        from ultralytics import YOLO
        model = YOLO('models/yolo11_dimension_safe.yaml')
        print("✅ Dimension-safe config works!")
        return True
    except Exception as e:
        print(f"❌ Dimension-safe config failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_progressive_config():
    """创建渐进式配置，逐步添加创新模块"""
    print("\n📈 Creating Progressive Config")
    print("="*40)
    
    # 基础配置 + 逐步添加创新模块
    progressive_config = """
# Progressive YOLO11 Config
# 渐进式配置，逐步集成创新模块

nc: 5
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]

backbone:
  # 标准结构
  - [-1, 1, Conv, [64, 3, 2]]     # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]    # 1-P2/4
  
  # 添加第一个创新：SE注意力
  - [-1, 1, SEAttention, []]      # 2 - SE attention
  - [-1, 2, C2f, [128, 256]]      # 3
  - [-1, 1, Conv, [256, 3, 2]]    # 4-P3/8
  
  # 添加第二个创新：蛇形卷积（保守使用）
  - [-1, 1, C3k2_SnakeDeformable, [256, 256]]  # 5 - Snake conv (same channels)
  - [-1, 1, C2f, [256, 512]]      # 6
  - [-1, 1, Conv, [512, 3, 2]]    # 7-P4/16
  
  # 添加第三个创新：CBAM注意力
  - [-1, 1, CBAM, []]             # 8 - CBAM attention
  - [-1, 2, C2f, [512, 512]]      # 9
  - [-1, 1, Conv, [512, 3, 2]]    # 10-P5/32
  
  # 标准结束
  - [-1, 2, C2f, [512, 1024]]     # 11
  - [-1, 1, SPPF, [1024, 1024]]   # 12

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 13
  - [[-1, 8], 1, Concat, [1]]                   # 14
  - [-1, 2, C2f, [1536, 512]]                   # 15
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 16
  - [[-1, 6], 1, Concat, [1]]                   # 17
  - [-1, 2, C2f, [1024, 256]]                   # 18
  
  # 添加微缺陷检测层
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 19
  - [[-1, 3], 1, Concat, [1]]                   # 20
  - [-1, 2, C2f, [512, 128]]                    # 21
  - [-1, 1, MicroDefectAttention, [128]]        # 22 - Micro defect attention
  
  # 重建检测层
  - [-1, 1, Conv, [128, 3, 2]]                  # 23
  - [[-1, 18], 1, Concat, [1]]                  # 24
  - [-1, 2, C2f, [384, 256]]                    # 25
  
  - [-1, 1, Conv, [256, 3, 2]]                  # 26
  - [[-1, 15], 1, Concat, [1]]                  # 27
  - [-1, 2, C2f, [768, 512]]                    # 28
  
  - [-1, 1, Conv, [512, 3, 2]]                  # 29
  - [[-1, 12], 1, Concat, [1]]                  # 30
  - [-1, 2, C2f, [1536, 1024]]                  # 31

  # 四尺度检测（包含P2小目标）
  - [[22, 25, 28, 31], 1, Detect, [nc]]         # 32
"""
    
    with open('models/yolo11_progressive.yaml', 'w') as f:
        f.write(progressive_config)
    
    print("💾 Created progressive config: models/yolo11_progressive.yaml")
    
    # 测试渐进式配置
    try:
        from ultralytics import YOLO
        model = YOLO('models/yolo11_progressive.yaml')
        print("✅ Progressive config works!")
        return True
    except Exception as e:
        print(f"❌ Progressive config failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_standard_yolo():
    """测试标准YOLO是否工作"""
    print("\n🧪 Testing Standard YOLO")
    print("="*30)
    
    try:
        from ultralytics import YOLO
        
        # 测试标准YOLO11n
        print("Testing yolo11n.yaml...")
        try:
            model = YOLO('yolo11n.yaml')
            print("✅ Standard yolo11n.yaml works")
            return True
        except Exception as e:
            print(f"❌ yolo11n.yaml failed: {e}")
        
        # 测试预训练模型
        print("Testing yolo11n.pt...")
        try:
            model = YOLO('yolo11n.pt')
            print("✅ Pretrained yolo11n.pt works")
            return True
        except Exception as e:
            print(f"❌ yolo11n.pt failed: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ Standard YOLO test failed: {e}")
        return False


def main():
    """主函数"""
    print("🐛 Tensor Size Debug Suite")
    print("="*60)
    
    # 1. 注册模块
    print("📦 Registering modules...")
    try:
        from register_modules import register_all_modules
        register_all_modules()
    except Exception as e:
        print(f"❌ Module registration failed: {e}")
        return False
    
    # 2. 测试标准YOLO
    standard_works = test_standard_yolo()
    
    if not standard_works:
        print("❌ Standard YOLO doesn't work. Environment issue.")
        return False
    
    # 3. 分析tensor流动
    analyze_tensor_flow()
    
    # 4. 创建并测试安全配置
    safe_success = create_dimension_safe_config()
    
    # 5. 创建并测试渐进式配置
    progressive_success = create_progressive_config()
    
    print("\n" + "="*60)
    print("🔍 Debug Results:")
    print(f"   Standard YOLO: {'✅' if standard_works else '❌'}")
    print(f"   Dimension-safe config: {'✅' if safe_success else '❌'}")
    print(f"   Progressive config: {'✅' if progressive_success else '❌'}")
    
    if safe_success or progressive_success:
        print("\n🎉 Found working configuration!")
        if safe_success:
            print("✅ Use: models/yolo11_dimension_safe.yaml")
        if progressive_success:
            print("✅ Use: models/yolo11_progressive.yaml")
    else:
        print("\n❌ All configurations failed.")
        print("🔧 Recommendation: Check ultralytics version and environment")
    
    return safe_success or progressive_success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)