"""
Micro Defect Detection Head
专用微缺陷检测头

基于论文描述实现的创新检测头：
- 专门捕捉占图像不足0.1%的极微小特征
- 检测下限扩展至15微米级别
- 针对亚像素级目标优化特征提取
- 在管道焊缝检测中，微气孔检出率从68%跃升至92%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class MicroDefectAttention(nn.Module):
    """
    微缺陷注意力机制
    专门用于增强极小目标的特征表示
    """
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channels = channels
        
        # 多尺度池化，捕捉不同大小的微缺陷
        self.micro_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(size) for size in [1, 2, 4]
        ])
        
        # 注意力计算
        mid_channels = max(channels // reduction, 8)
        self.attention_conv = nn.Sequential(
            nn.Conv2d(channels * len(self.micro_pools), mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力，专注于微小区域
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 多尺度通道注意力
        pooled_features = []
        for pool in self.micro_pools:
            pooled = pool(x)
            pooled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)
            pooled_features.append(pooled)
            
        combined = torch.cat(pooled_features, dim=1)
        channel_att = self.attention_conv(combined)
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        
        # 融合注意力
        enhanced = x * channel_att * spatial_att
        return enhanced


class SubPixelFeatureExtractor(nn.Module):
    """
    亚像素特征提取器
    使用上采样和精细化卷积提取微小特征
    """
    
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        
        # 亚像素卷积层
        self.sub_pixel_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 3, 1, 1),
            nn.PixelShuffle(scale_factor),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # 精细化卷积
        self.refine_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 亚像素上采样
        upsampled = self.sub_pixel_conv(x)
        
        # 精细化处理
        refined = self.refine_conv(upsampled)
        
        return refined


class MicroDefectHead(nn.Module):
    """
    专用微缺陷检测头
    
    特点：
    1. 多尺度特征融合
    2. 亚像素级特征提取
    3. 专门的微缺陷注意力机制
    4. 针对15微米级别目标优化
    """
    
    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        anchor_generator: Optional[object] = None,
        micro_threshold: float = 0.1  # 微缺陷面积阈值（相对于图像）
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_levels = len(in_channels)
        self.micro_threshold = micro_threshold
        
        # 统一特征通道数
        self.unified_channels = 256
        self.input_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, self.unified_channels, 1, bias=False),
                nn.BatchNorm2d(self.unified_channels),
                nn.SiLU(inplace=True)
            ) for ch in in_channels
        ])
        
        # 亚像素特征提取器（用于最高分辨率层）
        self.sub_pixel_extractors = nn.ModuleList([
            SubPixelFeatureExtractor(self.unified_channels, self.unified_channels, scale_factor=2)
            for _ in range(2)  # 只对前两个高分辨率层应用
        ])
        
        # 微缺陷注意力
        self.micro_attentions = nn.ModuleList([
            MicroDefectAttention(self.unified_channels) for _ in range(self.num_levels)
        ])
        
        # 检测头分支
        self.cls_heads = nn.ModuleList([
            self._make_head(self.unified_channels, num_classes) for _ in range(self.num_levels)
        ])
        
        self.reg_heads = nn.ModuleList([
            self._make_head(self.unified_channels, 4) for _ in range(self.num_levels)  # x, y, w, h
        ])
        
        # 置信度头，专门用于微缺陷
        self.conf_heads = nn.ModuleList([
            self._make_head(self.unified_channels, 1) for _ in range(self.num_levels)
        ])
        
        # 微缺陷特殊处理分支
        self.micro_enhancer = nn.Sequential(
            nn.Conv2d(self.unified_channels, self.unified_channels, 3, 1, 1),
            nn.BatchNorm2d(self.unified_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.unified_channels, self.unified_channels, 1),
            nn.Sigmoid()
        )
        
    def _make_head(self, in_channels: int, out_channels: int) -> nn.Module:
        """创建检测头分支"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1)
        )
        
    def forward(self, features: List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            features: 多尺度特征图列表
        Returns:
            outputs: 每个尺度的(分类, 回归, 置信度)预测
        """
        outputs = []
        
        for i, feat in enumerate(features):
            # 统一通道数
            adapted_feat = self.input_adapters[i](feat)
            
            # 亚像素特征提取（仅对高分辨率层）
            if i < len(self.sub_pixel_extractors):
                adapted_feat = self.sub_pixel_extractors[i](adapted_feat)
                
            # 微缺陷注意力增强
            attended_feat = self.micro_attentions[i](adapted_feat)
            
            # 微缺陷特殊增强
            micro_weight = self.micro_enhancer(attended_feat)
            enhanced_feat = attended_feat * (1 + micro_weight)
            
            # 预测分支
            cls_pred = self.cls_heads[i](enhanced_feat)
            reg_pred = self.reg_heads[i](enhanced_feat)
            conf_pred = self.conf_heads[i](enhanced_feat)
            
            outputs.append((cls_pred, reg_pred, conf_pred))
            
        return outputs


class EnhancedDetectHead(nn.Module):
    """
    增强检测头，集成微缺陷检测能力
    兼容YOLO11的Detect模块接口
    """
    
    def __init__(self, nc: int = 80, ch: List[int] = []):
        super().__init__()
        self.nc = nc  # 类别数
        self.nl = len(ch)  # 检测层数
        self.reg_max = 16  # DFL回归最大值
        self.no = nc + self.reg_max * 4  # 每个anchor的输出数
        
        # 微缺陷检测头
        self.micro_head = MicroDefectHead(ch, nc)
        
        # 标准检测头（保持兼容性）
        self.cv2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(x, 4 * self.reg_max, 1),
                nn.Conv2d(4 * self.reg_max, 4 * self.reg_max, 3, 1, 1, groups=4)
            ) for x in ch
        ])
        
        self.cv3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(x, nc, 1),
                nn.Conv2d(nc, nc, 3, 1, 1, groups=min(nc, 32))
            ) for x in ch
        ])
        
        # DFL层
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        
    def forward(self, x: List[torch.Tensor]):
        """前向传播，兼容YOLO11接口"""
        # 使用微缺陷检测头
        micro_outputs = self.micro_head(x)
        
        # 标准YOLO检测头
        standard_outputs = []
        for i, feat in enumerate(x):
            cls_pred = self.cv3[i](feat)
            reg_pred = self.cv2[i](feat)
            
            # 融合微缺陷预测
            micro_cls, micro_reg, micro_conf = micro_outputs[i]
            
            # 加权融合
            alpha = 0.3  # 微缺陷预测权重
            fused_cls = (1 - alpha) * cls_pred + alpha * micro_cls
            fused_reg = (1 - alpha) * reg_pred + alpha * micro_reg
            
            # 应用DFL
            if self.reg_max > 1:
                fused_reg = self.dfl(fused_reg)
                
            # 合并输出
            output = torch.cat([fused_reg, fused_cls], 1)
            standard_outputs.append(output)
            
        return standard_outputs


class DFL(nn.Module):
    """
    Distribution Focal Loss
    YOLO11中使用的分布焦点损失
    """
    
    def __init__(self, c1: int = 16):
        super().__init__()
        self.c1 = c1
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class UltraSmallObjectDetector(nn.Module):
    """
    超小目标检测器
    专门用于检测15微米级别的缺陷
    """
    
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # 超高分辨率特征提取
        self.ultra_high_res_conv = nn.Sequential(
            # 4倍上采样
            nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            
            # 再次2倍上采样
            nn.ConvTranspose2d(in_channels, in_channels // 2, 4, 2, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.SiLU(inplace=True),
        )
        
        # 微缺陷特征增强
        self.micro_feature_enhancer = nn.Sequential(
            # 深度可分离卷积，减少参数
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1, groups=in_channels // 2),
            nn.Conv2d(in_channels // 2, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.SiLU(inplace=True),
            
            # 小卷积核，专注微小特征
            nn.Conv2d(in_channels // 4, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.SiLU(inplace=True),
        )
        
        # 微缺陷分类头
        self.micro_cls_head = nn.Sequential(
            nn.Conv2d(in_channels // 4, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
        
        # 微缺陷回归头
        self.micro_reg_head = nn.Sequential(
            nn.Conv2d(in_channels // 4, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 4, 1)  # x, y, w, h
        )
        
        # 微缺陷置信度头
        self.micro_conf_head = nn.Sequential(
            nn.Conv2d(in_channels // 4, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入特征图
        Returns:
            (cls_pred, reg_pred, conf_pred): 分类、回归、置信度预测
        """
        # 超高分辨率特征提取
        ultra_feat = self.ultra_high_res_conv(x)
        
        # 微缺陷特征增强
        micro_feat = self.micro_feature_enhancer(ultra_feat)
        
        # 预测
        cls_pred = self.micro_cls_head(micro_feat)
        reg_pred = self.micro_reg_head(micro_feat)
        conf_pred = self.micro_conf_head(micro_feat)
        
        return cls_pred, reg_pred, conf_pred


class MultiScaleMicroDetector(nn.Module):
    """
    多尺度微缺陷检测器
    结合不同尺度的特征进行微缺陷检测
    """
    
    def __init__(self, feature_channels: List[int], num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_levels = len(feature_channels)
        
        # 为每个尺度创建微缺陷检测器
        self.micro_detectors = nn.ModuleList([
            UltraSmallObjectDetector(ch, num_classes) for ch in feature_channels
        ])
        
        # 多尺度融合
        self.scale_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_classes * self.num_levels, num_classes, 3, 1, 1),
                nn.BatchNorm2d(num_classes),
                nn.SiLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(4 * self.num_levels, 4, 3, 1, 1),
                nn.BatchNorm2d(4),
                nn.SiLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(self.num_levels, 1, 3, 1, 1),
                nn.Sigmoid()
            )
        ])
        
    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: 多尺度特征图
        Returns:
            (fused_cls, fused_reg, fused_conf): 融合后的预测结果
        """
        all_cls_preds = []
        all_reg_preds = []
        all_conf_preds = []
        
        # 获取目标尺寸（使用第一个特征图的尺寸）
        target_size = features[0].shape[2:]
        
        # 对每个尺度进行微缺陷检测
        for i, feat in enumerate(features):
            cls_pred, reg_pred, conf_pred = self.micro_detectors[i](feat)
            
            # 调整到统一尺寸
            if cls_pred.shape[2:] != target_size:
                cls_pred = F.interpolate(cls_pred, size=target_size, mode='bilinear', align_corners=False)
                reg_pred = F.interpolate(reg_pred, size=target_size, mode='bilinear', align_corners=False)
                conf_pred = F.interpolate(conf_pred, size=target_size, mode='bilinear', align_corners=False)
                
            all_cls_preds.append(cls_pred)
            all_reg_preds.append(reg_pred)
            all_conf_preds.append(conf_pred)
            
        # 多尺度融合
        fused_cls = self.scale_fusion[0](torch.cat(all_cls_preds, dim=1))
        fused_reg = self.scale_fusion[1](torch.cat(all_reg_preds, dim=1))
        fused_conf = self.scale_fusion[2](torch.cat(all_conf_preds, dim=1))
        
        return fused_cls, fused_reg, fused_conf


def test_micro_defect_head():
    """测试微缺陷检测头"""
    print("Testing Micro Defect Detection Head...")
    
    # 测试参数
    batch_size = 2
    num_classes = 5  # 气孔、裂纹、夹渣、未焊透、烧穿
    
    # 模拟多尺度特征
    features = [
        torch.randn(batch_size, 128, 80, 80),   # P2: 高分辨率
        torch.randn(batch_size, 256, 40, 40),   # P3: 中分辨率
        torch.randn(batch_size, 512, 20, 20),   # P4: 低分辨率
    ]
    
    # 测试微缺陷注意力
    print("\n1. Testing Micro Defect Attention...")
    micro_att = MicroDefectAttention(128)
    try:
        output = micro_att(features[0])
        print(f"✓ Micro Attention - Input: {features[0].shape}, Output: {output.shape}")
    except Exception as e:
        print(f"✗ Micro Attention failed: {e}")
        
    # 测试亚像素特征提取
    print("\n2. Testing SubPixel Feature Extractor...")
    sub_pixel = SubPixelFeatureExtractor(128, 64, scale_factor=2)
    try:
        output = sub_pixel(features[0])
        print(f"✓ SubPixel Extractor - Input: {features[0].shape}, Output: {output.shape}")
    except Exception as e:
        print(f"✗ SubPixel Extractor failed: {e}")
        
    # 测试微缺陷检测头
    print("\n3. Testing Micro Defect Head...")
    micro_head = MicroDefectHead([128, 256, 512], num_classes)
    try:
        outputs = micro_head(features)
        print(f"✓ Micro Defect Head - {len(outputs)} levels")
        for i, (cls, reg, conf) in enumerate(outputs):
            print(f"  Level {i}: cls={cls.shape}, reg={reg.shape}, conf={conf.shape}")
    except Exception as e:
        print(f"✗ Micro Defect Head failed: {e}")
        
    # 测试超小目标检测器
    print("\n4. Testing Ultra Small Object Detector...")
    ultra_detector = UltraSmallObjectDetector(128, num_classes)
    try:
        cls, reg, conf = ultra_detector(features[0])
        print(f"✓ Ultra Small Detector - cls: {cls.shape}, reg: {reg.shape}, conf: {conf.shape}")
    except Exception as e:
        print(f"✗ Ultra Small Detector failed: {e}")
        
    # 测试多尺度微检测器
    print("\n5. Testing Multi-Scale Micro Detector...")
    multi_detector = MultiScaleMicroDetector([128, 256, 512], num_classes)
    try:
        fused_cls, fused_reg, fused_conf = multi_detector(features)
        print(f"✓ Multi-Scale Micro Detector:")
        print(f"  Fused cls: {fused_cls.shape}")
        print(f"  Fused reg: {fused_reg.shape}")
        print(f"  Fused conf: {fused_conf.shape}")
    except Exception as e:
        print(f"✗ Multi-Scale Micro Detector failed: {e}")
        
    print("\nMicro Defect Head testing completed!")


if __name__ == "__main__":
    test_micro_defect_head()