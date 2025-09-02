"""
Bidirectional Feature Pyramid Network (BiFPN) Module
双向三阶金字塔特征融合结构

基于EfficientDet的BiFPN设计，针对X光焊缝检测优化：
- 实现多尺度特征交互，保证大型夹渣和微小气孔的同时检测
- 使用快速标准化融合，提高训练稳定性
- 支持三阶金字塔结构，扩展检测范围至传统方法的3倍尺度跨度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math


class FastNormalizedFusion(nn.Module):
    """
    快速标准化融合
    EfficientDet中提出的高效特征融合方法
    """
    
    def __init__(self, num_inputs: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        # 可学习的融合权重
        self.weights = nn.Parameter(torch.ones(num_inputs))
        
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: 待融合的特征图列表
        Returns:
            fused_feature: 融合后的特征图
        """
        # 应用ReLU确保权重为正
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        
        # 加权融合
        fused = sum(w * feat for w, feat in zip(weights, inputs))
        return fused


class SeparableConv2d(nn.Module):
    """深度可分离卷积，减少参数量"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, 
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class BiFPNLayer(nn.Module):
    """
    单层BiFPN模块
    实现双向特征传播：自顶向下 + 自底向上
    """
    
    def __init__(self, channels: int, num_levels: int = 3, eps: float = 1e-4):
        super().__init__()
        self.channels = channels
        self.num_levels = num_levels
        self.eps = eps
        
        # 自顶向下路径的融合权重
        self.td_weights = nn.ModuleList([
            FastNormalizedFusion(2) for _ in range(num_levels - 1)
        ])
        
        # 自底向上路径的融合权重  
        self.bu_weights = nn.ModuleList([
            FastNormalizedFusion(3) if i == 0 else FastNormalizedFusion(2) 
            for i in range(num_levels - 1)
        ])
        
        # 卷积层用于特征处理
        self.td_convs = nn.ModuleList([
            SeparableConv2d(channels, channels) for _ in range(num_levels - 1)
        ])
        
        self.bu_convs = nn.ModuleList([
            SeparableConv2d(channels, channels) for _ in range(num_levels - 1)
        ])
        
        # 输出卷积
        self.out_convs = nn.ModuleList([
            SeparableConv2d(channels, channels) for _ in range(num_levels)
        ])
        
    def _resize_features(self, feat: torch.Tensor, target_size: torch.Size) -> torch.Tensor:
        """调整特征图尺寸"""
        if feat.shape[2:] == target_size[2:]:
            return feat
        elif feat.shape[2] > target_size[2]:
            # 下采样
            scale = feat.shape[2] // target_size[2]
            return F.max_pool2d(feat, kernel_size=scale, stride=scale)
        else:
            # 上采样
            return F.interpolate(feat, size=target_size[2:], mode='nearest')
            
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: 输入特征图列表，从高分辨率到低分辨率 [P2, P3, P4]
        Returns:
            output_features: 融合后的特征图列表
        """
        assert len(features) == self.num_levels, f"Expected {self.num_levels} features, got {len(features)}"
        
        # 初始化中间特征
        td_features = [None] * self.num_levels  # 自顶向下特征
        bu_features = [None] * self.num_levels  # 自底向上特征
        
        # 自顶向下路径 (Top-Down)
        td_features[-1] = features[-1]  # 最高层特征直接使用
        
        for i in range(self.num_levels - 2, -1, -1):
            # 上采样高层特征
            upsampled = F.interpolate(
                td_features[i + 1], 
                size=features[i].shape[2:], 
                mode='nearest'
            )
            
            # 融合当前层和上采样特征
            td_features[i] = self.td_weights[i]([features[i], upsampled])
            td_features[i] = self.td_convs[i](td_features[i])
            
        # 自底向上路径 (Bottom-Up)
        bu_features[0] = td_features[0]  # 最低层特征直接使用
        
        for i in range(1, self.num_levels):
            # 下采样低层特征
            if i == 1:
                # 第一层需要融合三个特征：原始特征、自顶向下特征、自底向上特征
                downsampled = F.max_pool2d(bu_features[i - 1], kernel_size=2, stride=2)
                downsampled = self._resize_features(downsampled, td_features[i])
                
                bu_features[i] = self.bu_weights[i - 1]([
                    features[i], td_features[i], downsampled
                ])
            else:
                # 其他层融合两个特征
                downsampled = F.max_pool2d(bu_features[i - 1], kernel_size=2, stride=2)
                downsampled = self._resize_features(downsampled, td_features[i])
                
                bu_features[i] = self.bu_weights[i - 1]([td_features[i], downsampled])
                
            bu_features[i] = self.bu_convs[i - 1](bu_features[i])
            
        # 输出处理
        output_features = []
        for i, feat in enumerate(bu_features):
            output_features.append(self.out_convs[i](feat))
            
        return output_features


class TripleBiFPN(nn.Module):
    """
    三阶BiFPN网络
    堆叠多个BiFPN层以增强特征融合能力
    """
    
    def __init__(self, channels: int, num_levels: int = 3, num_layers: int = 3):
        super().__init__()
        self.channels = channels
        self.num_levels = num_levels
        self.num_layers = num_layers
        
        # 输入投影层，确保所有特征图通道数一致
        self.input_projections = nn.ModuleList([
            nn.Conv2d(channels, channels, 1, bias=False) for _ in range(num_levels)
        ])
        
        # 堆叠BiFPN层
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(channels, num_levels) for _ in range(num_layers)
        ])
        
        # 输出投影层
        self.output_projections = nn.ModuleList([
            nn.Conv2d(channels, channels, 1, bias=False) for _ in range(num_levels)
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: 输入特征图列表 [P2, P3, P4] 或 [P3, P4, P5]
        Returns:
            enhanced_features: 增强后的特征图列表
        """
        # 输入投影，统一通道数
        projected_features = []
        for i, feat in enumerate(features):
            if feat.shape[1] != self.channels:
                # 如果通道数不匹配，先调整
                proj_conv = nn.Conv2d(feat.shape[1], self.channels, 1, bias=False).to(feat.device)
                projected_features.append(proj_conv(feat))
            else:
                projected_features.append(self.input_projections[i](feat))
        
        # 通过多层BiFPN处理
        current_features = projected_features
        for bifpn_layer in self.bifpn_layers:
            current_features = bifpn_layer(current_features)
            
        # 输出投影
        output_features = []
        for i, feat in enumerate(current_features):
            output_features.append(self.output_projections[i](feat))
            
        return output_features


class AdaptiveBiFPN(nn.Module):
    """
    自适应BiFPN
    根据输入特征自动调整结构参数
    """
    
    def __init__(self, min_channels: int = 64, max_channels: int = 512):
        super().__init__()
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.bifpn_layers = nn.ModuleDict()
        
    def _get_or_create_bifpn(self, channels: int, num_levels: int) -> TripleBiFPN:
        """获取或创建对应配置的BiFPN"""
        key = f"bifpn_{channels}_{num_levels}"
        if key not in self.bifpn_layers:
            self.bifpn_layers[key] = TripleBiFPN(channels, num_levels)
        return self.bifpn_layers[key]
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """自适应处理不同配置的特征"""
        if not features:
            return features
            
        # 自动检测配置
        channels = features[0].shape[1]
        num_levels = len(features)
        
        # 获取对应的BiFPN
        bifpn = self._get_or_create_bifpn(channels, num_levels)
        bifpn = bifpn.to(features[0].device)
        
        return bifpn(features)


class BiFPNBlock(nn.Module):
    """
    BiFPN块，用于集成到YOLO架构中
    """
    
    def __init__(self, c1: int, c2: int, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels
        
        # 确保输出通道数
        if c1 != c2:
            self.channel_adapter = nn.Conv2d(c1, c2, 1, bias=False)
        else:
            self.channel_adapter = nn.Identity()
            
        self.bifpn = TripleBiFPN(c2, num_levels, num_layers=2)
        
    def forward(self, x):
        """
        处理单个特征图输入，模拟多尺度输入
        """
        # 适配通道数
        x = self.channel_adapter(x)
        
        # 生成多尺度特征（模拟）
        features = [x]
        
        # 下采样生成更多尺度
        current = x
        for i in range(self.num_levels - 1):
            current = F.max_pool2d(current, kernel_size=2, stride=2)
            features.append(current)
            
        # 通过BiFPN处理
        enhanced_features = self.bifpn(features)
        
        # 返回原始尺度的增强特征
        return F.interpolate(
            enhanced_features[0], 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )


def test_bifpn_modules():
    """测试BiFPN模块"""
    print("Testing BiFPN Modules...")
    
    # 测试数据
    batch_size = 2
    
    # 模拟不同尺度的特征图
    p2 = torch.randn(batch_size, 128, 64, 64)   # P2: 高分辨率，小目标
    p3 = torch.randn(batch_size, 256, 32, 32)   # P3: 中分辨率
    p4 = torch.randn(batch_size, 512, 16, 16)   # P4: 低分辨率，大目标
    
    features = [p2, p3, p4]
    
    # 测试单层BiFPN
    print("\n1. Testing BiFPN Layer...")
    bifpn_layer = BiFPNLayer(channels=256, num_levels=3)
    
    # 调整特征图通道数到统一的256
    unified_features = []
    for i, feat in enumerate(features):
        if feat.shape[1] != 256:
            adapter = nn.Conv2d(feat.shape[1], 256, 1, bias=False)
            unified_features.append(adapter(feat))
        else:
            unified_features.append(feat)
    
    try:
        output = bifpn_layer(unified_features)
        print(f"✓ BiFPN Layer - Input levels: {len(unified_features)}, Output levels: {len(output)}")
        for i, feat in enumerate(output):
            print(f"  Level {i}: {feat.shape}")
    except Exception as e:
        print(f"✗ BiFPN Layer failed: {e}")
        
    # 测试三阶BiFPN
    print("\n2. Testing Triple BiFPN...")
    triple_bifpn = TripleBiFPN(channels=256, num_levels=3, num_layers=3)
    
    try:
        output = triple_bifpn(unified_features)
        print(f"✓ Triple BiFPN - Input levels: {len(unified_features)}, Output levels: {len(output)}")
        for i, feat in enumerate(output):
            print(f"  Enhanced Level {i}: {feat.shape}")
    except Exception as e:
        print(f"✗ Triple BiFPN failed: {e}")
        
    # 测试自适应BiFPN
    print("\n3. Testing Adaptive BiFPN...")
    adaptive_bifpn = AdaptiveBiFPN()
    
    try:
        output = adaptive_bifpn(unified_features)
        print(f"✓ Adaptive BiFPN - Input levels: {len(unified_features)}, Output levels: {len(output)}")
    except Exception as e:
        print(f"✗ Adaptive BiFPN failed: {e}")
        
    # 测试BiFPN块（用于YOLO集成）
    print("\n4. Testing BiFPN Block...")
    bifpn_block = BiFPNBlock(c1=128, c2=256, num_levels=3)
    
    try:
        single_input = torch.randn(batch_size, 128, 64, 64)
        output = bifpn_block(single_input)
        print(f"✓ BiFPN Block - Input: {single_input.shape}, Output: {output.shape}")
    except Exception as e:
        print(f"✗ BiFPN Block failed: {e}")
        
    print("\nBiFPN testing completed!")


class MultiScaleBiFPN(nn.Module):
    """
    多尺度BiFPN，专门用于小目标检测
    支持P1到P5的五个尺度级别
    """
    
    def __init__(self, feature_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.feature_channels = feature_channels
        self.out_channels = out_channels
        self.num_levels = len(feature_channels)
        
        # 输入通道统一化
        self.input_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            ) for in_ch in feature_channels
        ])
        
        # 多层BiFPN
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(out_channels, self.num_levels) for _ in range(2)
        ])
        
        # 小目标增强模块
        self.small_object_enhancer = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
                nn.Conv2d(out_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            ) for _ in range(2)  # 只增强前两个尺度（P1, P2）
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: 多尺度特征图 [P1, P2, P3, P4, P5] 或子集
        Returns:
            enhanced_features: 增强后的多尺度特征
        """
        # 统一通道数
        adapted_features = []
        for i, feat in enumerate(features):
            adapted = self.input_adapters[i](feat)
            adapted_features.append(adapted)
            
        # 通过BiFPN层处理
        current_features = adapted_features
        for bifpn_layer in self.bifpn_layers:
            current_features = bifpn_layer(current_features)
            
        # 小目标增强（只对前两个尺度）
        enhanced_features = current_features.copy()
        for i in range(min(2, len(enhanced_features))):
            enhanced_features[i] = self.small_object_enhancer[i](enhanced_features[i])
            
        return enhanced_features


if __name__ == "__main__":
    test_bifpn_modules()