"""
Core Modules Final - 最终核心模块
专注于四大功能，避免所有尺寸和断言问题
基于标准YOLO模块设计，确保兼容性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, C2f
import math


class MicroDefectHead(nn.Module):
    """
    专用微缺陷检测头 - 功能1
    在P1/P2高分辨率层检测15微米级别缺陷
    """
    def __init__(self, c1, c2, reduction=4):
        super().__init__()
        # 确保输入输出通道匹配
        self.adapt = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        
        # 微缺陷特征增强
        self.enhance = nn.Sequential(
            Conv(c2, c2, 3, 1, 1),  # 3x3卷积
            Conv(c2, c2, 1, 1),     # 1x1卷积
        )
        
        # 通道注意力 - 专门用于微小目标
        mid_channels = max(c2 // reduction, 16)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, c2, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力 - 专门用于微小区域
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 通道适配
        x = self.adapt(x)
        
        # 特征增强
        enhanced = self.enhance(x)
        
        # 通道注意力
        ca = self.channel_att(enhanced)
        enhanced = enhanced * ca
        
        # 空间注意力
        avg_out = torch.mean(enhanced, dim=1, keepdim=True)
        max_out, _ = torch.max(enhanced, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        output = enhanced * sa
        
        return output


class SnakeConv(nn.Module):
    """
    蛇形卷积 - 功能2 简化版
    保持标准卷积接口，避免复杂的可变形实现
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
            
        # 标准卷积作为基础
        self.conv = Conv(c1, c2, k, s, p, g, act)
        
        # 蛇形感受野模拟 - 使用多个小卷积核
        self.snake_convs = nn.ModuleList([
            nn.Conv2d(c1, c2 // 4, 1, s, 0, bias=False),  # 中心点
            nn.Conv2d(c1, c2 // 4, (1, 3), s, (0, 1), bias=False),  # 水平
            nn.Conv2d(c1, c2 // 4, (3, 1), s, (1, 0), bias=False),  # 垂直
            nn.Conv2d(c1, c2 // 4, 3, s, 1, bias=False),  # 对角
        ])
        
        self.fusion = Conv(c2, c2, 1, 1, act=False)
        self.act = nn.SiLU() if act else nn.Identity()
        
    def forward(self, x):
        # 标准卷积
        main_out = self.conv(x)
        
        # 蛇形卷积组合
        snake_outs = [conv(x) for conv in self.snake_convs]
        snake_combined = torch.cat(snake_outs, dim=1)
        snake_out = self.fusion(snake_combined)
        
        # 加权融合
        alpha = 0.3  # 蛇形卷积权重
        output = (1 - alpha) * main_out + alpha * snake_out
        
        return self.act(output)


class BiFPNSimple(nn.Module):
    """
    简化BiFPN - 功能3
    避免复杂的多输入处理
    """
    def __init__(self, c1, c2):
        super().__init__()
        # 通道适配
        self.adapt = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        
        # 多尺度处理
        self.scale_convs = nn.ModuleList([
            Conv(c2, c2, 3, 1, 1),  # 原始尺度
            Conv(c2, c2, 3, 1, 2),  # 扩张卷积1
            Conv(c2, c2, 3, 1, 3),  # 扩张卷积2
        ])
        
        # 融合权重
        self.fusion_weights = nn.Parameter(torch.ones(3))
        
        # 输出处理
        self.output_conv = Conv(c2, c2, 1, 1)
        
    def forward(self, x):
        x = self.adapt(x)
        
        # 多尺度特征提取
        scale_features = []
        for conv in self.scale_convs:
            scale_features.append(conv(x))
        
        # 加权融合
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = sum(w * feat for w, feat in zip(weights, scale_features))
        
        # 输出处理
        output = self.output_conv(fused)
        
        return output


class EnhancedC2f(nn.Module):
    """
    增强C2f - 集成蛇形卷积
    基于标准C2f，确保兼容性
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 输出卷积
        
        # 混合使用标准卷积和蛇形卷积
        self.m = nn.ModuleList([
            SnakeConv(self.c, self.c, 3, 1, 1) if i == 0  # 第一层使用蛇形卷积
            else Conv(self.c, self.c, 3, 1, 1)             # 其他层使用标准卷积
            for i in range(n)
        ])
        
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SmallObjectAttention(nn.Module):
    """
    小目标注意力 - 避免c1==c2断言问题
    专门用于小目标检测优化
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        # 避免断言问题，不要求输入输出通道相等
        mid_channels = max(channels // reduction, 8)
        
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 通道注意力
        ca = self.channel_att(x)
        x = x * ca
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        
        return x


def register_core_modules():
    """注册核心模块"""
    try:
        import ultralytics.nn.tasks as tasks
        import ultralytics.nn.modules as modules
        
        # 注册四大功能模块
        module_dict = {
            'MicroDefectHead': MicroDefectHead,           # 功能1: 专用微缺陷检测头
            'SnakeConv': SnakeConv,                       # 功能2: 蛇形卷积
            'BiFPNSimple': BiFPNSimple,                   # 功能3: 简化BiFPN
            'EnhancedC2f': EnhancedC2f,                   # 集成蛇形卷积的C2f
            'SmallObjectAttention': SmallObjectAttention, # 小目标注意力
        }
        
        # 注册到两个命名空间
        for name, module_class in module_dict.items():
            setattr(tasks, name, module_class)
            setattr(modules, name, module_class)
        
        print("✅ Core modules registered successfully!")
        print("   Four core functions:")
        print("   1. MicroDefectHead - 专用微缺陷检测头")
        print("   2. SnakeConv - 蛇形卷积")
        print("   3. BiFPNSimple - 简化BiFPN")
        print("   4. EnhancedC2f - 增强C2f (集成蛇形卷积)")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to register core modules: {e}")
        return False


# 自动注册
register_core_modules()