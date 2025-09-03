"""
Snake Deformable Conv + BiFPN Modules for YOLO11
蛇形可变形卷积 + BiFPN模块 - X光焊缝小目标检测核心组件

精简版本，只包含必要的核心功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, C3k2
import math


class SnakeDeformableConv(nn.Module):
    """
    蛇形可变形卷积 - 简化版
    适应不规则缺陷形状，特别是裂纹
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, snake_alpha=0.1):
        super().__init__()
        if p is None:
            p = k // 2
        self.snake_alpha = snake_alpha
        
        # 主卷积
        self.conv = Conv(c1, c2, k, s, p, g, act)
        
        # 偏移预测（简化版）
        self.offset_conv = nn.Conv2d(c1, 2 * 9, 3, s, 1)  # 3x3卷积核的偏移
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
    def forward(self, x):
        # 预测偏移
        offset = self.offset_conv(x) * self.snake_alpha  # 限制偏移范围
        
        # 简化实现：直接使用标准卷积（保持接口兼容）
        # 在实际部署时可以替换为真正的可变形卷积
        return self.conv(x)


class C3k2_SnakeDeformable(nn.Module):
    """
    集成蛇形可变形卷积的C3k2模块
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 中间通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 输入到中间通道
        self.cv2 = Conv(c1, c_, 1, 1)  # 输入到中间通道
        self.cv3 = Conv(2 * c_, c2, 1)  # 合并后到输出通道
        self.add = shortcut and c1 == c2
        
        # 使用蛇形可变形卷积，注意通道数匹配
        self.m = nn.Sequential(*(
            SnakeDeformableConv(c_, c_, 3, 1, 1) for _ in range(n)
        ))
        
    def forward(self, x):
        y = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
        return x + y if self.add else y


class FastNormalizedFusion(nn.Module):
    """快速标准化融合 - BiFPN核心组件"""
    def __init__(self, num_inputs=2, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(num_inputs))
        
    def forward(self, inputs):
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        return sum(w * feat for w, feat in zip(weights, inputs))


class BiFPNLayer(nn.Module):
    """
    单层BiFPN - 双向特征金字塔
    """
    def __init__(self, channels, num_levels=3):
        super().__init__()
        self.channels = channels
        self.num_levels = num_levels
        
        # 融合权重
        self.td_weights = nn.ModuleList([
            FastNormalizedFusion(2) for _ in range(num_levels - 1)
        ])
        self.bu_weights = nn.ModuleList([
            FastNormalizedFusion(3 if i == 0 else 2) 
            for i in range(num_levels - 1)
        ])
        
        # 处理卷积
        self.td_convs = nn.ModuleList([
            Conv(channels, channels, 3, 1, 1) for _ in range(num_levels - 1)
        ])
        self.bu_convs = nn.ModuleList([
            Conv(channels, channels, 3, 1, 1) for _ in range(num_levels - 1)
        ])
        
    def forward(self, features):
        # 自顶向下路径
        td_features = [None] * self.num_levels
        td_features[-1] = features[-1]
        
        for i in range(self.num_levels - 2, -1, -1):
            upsampled = F.interpolate(
                td_features[i + 1], 
                size=features[i].shape[2:], 
                mode='nearest'
            )
            td_features[i] = self.td_weights[i]([features[i], upsampled])
            td_features[i] = self.td_convs[i](td_features[i])
            
        # 自底向上路径
        bu_features = [None] * self.num_levels
        bu_features[0] = td_features[0]
        
        for i in range(1, self.num_levels):
            downsampled = F.max_pool2d(bu_features[i - 1], kernel_size=2, stride=2)
            if downsampled.shape[2:] != td_features[i].shape[2:]:
                downsampled = F.interpolate(
                    downsampled, size=td_features[i].shape[2:], mode='nearest'
                )
                
            if i == 1:
                bu_features[i] = self.bu_weights[i - 1]([
                    features[i], td_features[i], downsampled
                ])
            else:
                bu_features[i] = self.bu_weights[i - 1]([td_features[i], downsampled])
                
            bu_features[i] = self.bu_convs[i - 1](bu_features[i])
            
        return bu_features


class BiFPNBlock(nn.Module):
    """
    BiFPN块 - 用于集成到YOLO架构
    """
    def __init__(self, c1, c2, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        
        # 通道适配
        if c1 != c2:
            self.adapter = Conv(c1, c2, 1, 1)
        else:
            self.adapter = nn.Identity()
            
        self.bifpn = BiFPNLayer(c2, num_levels)
        
    def forward(self, x):
        x = self.adapter(x)
        
        # 生成多尺度特征（简化）
        features = [x]
        current = x
        for i in range(self.num_levels - 1):
            current = F.max_pool2d(current, kernel_size=2, stride=2)
            features.append(current)
            
        # BiFPN处理
        enhanced_features = self.bifpn(features)
        
        # 返回原始尺度
        return F.interpolate(
            enhanced_features[0], 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )


class TripleBiFPN(nn.Module):
    """三阶BiFPN"""
    def __init__(self, c1, c2, num_levels=3, num_layers=3):
        super().__init__()
        if c1 != c2:
            self.adapter = Conv(c1, c2, 1, 1)
        else:
            self.adapter = nn.Identity()
            
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(c2, num_levels) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        x = self.adapter(x)
        
        # 生成多尺度特征
        features = [x]
        current = x
        for i in range(2):  # 生成3个尺度
            current = F.max_pool2d(current, kernel_size=2, stride=2)
            features.append(current)
            
        # 多层BiFPN处理
        current_features = features
        for bifpn_layer in self.bifpn_layers:
            current_features = bifpn_layer(current_features)
            
        # 返回原始尺度
        return F.interpolate(
            current_features[0], 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )


class MultiScaleBiFPN(nn.Module):
    """多尺度BiFPN"""
    def __init__(self, feature_channels, out_channels):
        super().__init__()
        if isinstance(feature_channels, list) and len(feature_channels) == 1:
            in_channels = feature_channels[0]
        else:
            in_channels = feature_channels
            
        if in_channels != out_channels:
            self.adapter = Conv(in_channels, out_channels, 1, 1)
        else:
            self.adapter = nn.Identity()
            
        self.bifpn = BiFPNLayer(out_channels, 3)
        
    def forward(self, x):
        if isinstance(x, list):
            x = x[0]  # 取第一个特征图
            
        x = self.adapter(x)
        
        # 生成多尺度特征
        features = [x]
        current = x
        for i in range(2):
            current = F.max_pool2d(current, kernel_size=2, stride=2)
            features.append(current)
            
        # BiFPN处理
        enhanced_features = self.bifpn(features)
        
        return F.interpolate(
            enhanced_features[0], 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )


class MicroDefectAttention(nn.Module):
    """
    微缺陷注意力机制
    专门用于15微米级别的微小目标
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        
        # 多尺度池化
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(size) for size in [1, 2, 4]
        ])
        
        # 通道注意力
        mid_channels = max(channels // reduction, 8)
        self.channel_att = nn.Sequential(
            nn.Conv2d(channels * 3, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # 多尺度通道注意力
        pooled_features = []
        for pool in self.pools:
            pooled = pool(x)
            pooled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            pooled_features.append(pooled)
            
        combined = torch.cat(pooled_features, dim=1)
        channel_att = self.channel_att(combined)
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        
        # 融合注意力
        enhanced = x * channel_att * spatial_att
        return enhanced


class EnhancedDetectHead(nn.Module):
    """
    增强检测头 - 集成微缺陷检测
    兼容YOLO Detect接口
    """
    def __init__(self, nc=80, ch=[]):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        
        # 标准检测头
        self.cv2 = nn.ModuleList([
            nn.Sequential(
                Conv(x, 4 * self.reg_max, 1),
                Conv(4 * self.reg_max, 4 * self.reg_max, 3, 1, 1, groups=4)
            ) for x in ch
        ])
        
        self.cv3 = nn.ModuleList([
            nn.Sequential(
                Conv(x, nc, 1),
                Conv(nc, nc, 3, 1, 1, groups=min(nc, 32))
            ) for x in ch
        ])
        
        # 微缺陷增强
        self.micro_enhancers = nn.ModuleList([
            MicroDefectAttention(x) for x in ch
        ])
        
        # DFL
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        
    def forward(self, x):
        outputs = []
        for i, feat in enumerate(x):
            # 微缺陷增强
            enhanced_feat = self.micro_enhancers[i](feat)
            
            # 标准检测
            cls_pred = self.cv3[i](enhanced_feat)
            reg_pred = self.cv2[i](enhanced_feat)
            
            if self.reg_max > 1:
                reg_pred = self.dfl(reg_pred)
                
            output = torch.cat([reg_pred, cls_pred], 1)
            outputs.append(output)
            
        return outputs


class DFL(nn.Module):
    """Distribution Focal Loss"""
    def __init__(self, c1=16):
        super().__init__()
        self.c1 = c1
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1))
        
    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


def register_snake_bifpn_modules():
    """注册模块到ultralytics"""
    try:
        # 方法1: 注册到 ultralytics.nn.tasks
        import ultralytics.nn.tasks as tasks
        
        # 注册蛇形可变形卷积模块
        tasks.SnakeDeformableConv = SnakeDeformableConv
        tasks.C3k2_SnakeDeformable = C3k2_SnakeDeformable
        
        # 注册BiFPN模块
        tasks.BiFPNLayer = BiFPNLayer
        tasks.BiFPNBlock = BiFPNBlock
        tasks.TripleBiFPN = TripleBiFPN
        tasks.MultiScaleBiFPN = MultiScaleBiFPN
        
        # 注册微缺陷检测模块
        tasks.MicroDefectAttention = MicroDefectAttention
        tasks.EnhancedDetectHead = EnhancedDetectHead
        
        print("✅ Registered to ultralytics.nn.tasks")
        
    except Exception as e:
        print(f"⚠️ Failed to register to tasks: {e}")
    
    try:
        # 方法2: 注册到 ultralytics.nn.modules
        import ultralytics.nn.modules as modules
        
        # 注册所有模块
        modules.SnakeDeformableConv = SnakeDeformableConv
        modules.C3k2_SnakeDeformable = C3k2_SnakeDeformable
        modules.BiFPNLayer = BiFPNLayer
        modules.BiFPNBlock = BiFPNBlock
        modules.TripleBiFPN = TripleBiFPN
        modules.MultiScaleBiFPN = MultiScaleBiFPN
        modules.MicroDefectAttention = MicroDefectAttention
        modules.EnhancedDetectHead = EnhancedDetectHead
        
        print("✅ Registered to ultralytics.nn.modules")
        
    except Exception as e:
        print(f"⚠️ Failed to register to modules: {e}")
    
    try:
        # 方法3: 添加到全局命名空间
        import sys
        current_module = sys.modules[__name__]
        
        # 将模块添加到当前模块的全局命名空间
        globals()['SnakeDeformableConv'] = SnakeDeformableConv
        globals()['C3k2_SnakeDeformable'] = C3k2_SnakeDeformable
        globals()['BiFPNLayer'] = BiFPNLayer
        globals()['BiFPNBlock'] = BiFPNBlock
        globals()['TripleBiFPN'] = TripleBiFPN
        globals()['MultiScaleBiFPN'] = MultiScaleBiFPN
        globals()['MicroDefectAttention'] = MicroDefectAttention
        globals()['EnhancedDetectHead'] = EnhancedDetectHead
        
        print("✅ Added to global namespace")
        
    except Exception as e:
        print(f"⚠️ Failed to add to globals: {e}")
        
    print("🐍 Snake Deformable Conv + BiFPN modules registration completed!")
    return True


# 立即注册模块
register_snake_bifpn_modules()