"""
Custom PyTorch modules for enhanced YOLO11 surface defect detection.
This module contains attention mechanisms and other custom layers to improve MAP values.
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, C3k2, C2f
from ultralytics.utils.torch_utils import fuse_conv_and_bn


class CoordAtt(nn.Module):
    """
    Coordinate Attention module for improved spatial feature representation.
    
    This attention mechanism captures position-sensitive information along both
    horizontal and vertical directions, which is particularly beneficial for
    surface defect detection where spatial relationships are crucial.
    
    Args:
        inp (int): Input channel dimension
        reduction (int): Channel reduction ratio for efficiency (default: 32)
    """
    
    def __init__(self, inp, reduction=32):
        super().__init__()
        # 确保最小通道数满足reduction要求
        self.reduction = max(reduction, 1)  # 防止除零错误
        mip = max(8, inp // self.reduction)
        
        # 1. 坐标编码层
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))    # [B,C,H,1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))    # [B,C,1,W]
        
        # 2. 共享卷积层 (修正通道维度)
        self.conv1 = nn.Conv2d(inp, mip, 1, bias=False)  # 输入通道=inp
        self.bn1 = nn.BatchNorm2d(mip)  # 添加BN稳定训练
        self.act = nn.Hardswish()  # 替换自定义h_sigmoid
        
        # 3. 方向注意力卷积
        self.conv_h = nn.Conv2d(mip, inp, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, inp, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.shape
        
        # 水平方向编码
        x_h = self.pool_h(x)  # [B,C,H,1]
        # 垂直方向编码 (无需permute)
        x_w = self.pool_w(x)  # [B,C,1,W]
        
        # 分别处理两个方向
        y_h = self.act(self.bn1(self.conv1(x_h)))  # [B, mip, H, 1]
        y_w = self.act(self.bn1(self.conv1(x_w)))  # [B, mip, 1, W]
        
        # 生成注意力权重
        att_h = self.sigmoid(self.conv_h(y_h))  # [B,C,H,1]
        att_w = self.sigmoid(self.conv_w(y_w))  # [B,C,1,W]
        
        # 空间加权
        return identity * att_h * att_w  # [B,C,H,W]


class C3k2_CoordAtt(C3k2):
    """
    Enhanced C3k2 module with Coordinate Attention for better feature representation.
    This combines the efficient C3k2 architecture with coordinate attention mechanism.
    """
    
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        self.coord_att = CoordAtt(c2)
    
    def forward(self, x):
        x = super().forward(x)
        return self.coord_att(x)


class C2f_CoordAtt(C2f):
    """
    Enhanced C2f module with Coordinate Attention for better feature representation.
    This combines the C2f architecture with coordinate attention mechanism.
    """
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.coord_att = CoordAtt(c2)
    
    def forward(self, x):
        x = super().forward(x)
        return self.coord_att(x)


class EnhancedConv(Conv):
    """
    Enhanced Conv module with Coordinate Attention.
    Useful for adding attention to key convolution layers in the backbone.
    """
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.coord_att = CoordAtt(c2)
    
    def forward(self, x):
        x = super().forward(x)
        return self.coord_att(x)


def register_custom_modules():
    """
    Register custom modules with ultralytics framework.
    This function should be called before creating YOLO models that use these modules.
    """
    import ultralytics.nn.tasks as tasks_module
    
    # Add custom modules to the tasks module's global namespace
    # This is how ultralytics resolves custom modules during model parsing
    tasks_module.CoordAtt = CoordAtt
    tasks_module.C3k2_CoordAtt = C3k2_CoordAtt
    tasks_module.C2f_CoordAtt = C2f_CoordAtt
    tasks_module.EnhancedConv = EnhancedConv
    
    print("Custom modules registered successfully:")
    print("- CoordAtt: Coordinate Attention mechanism")
    print("- C3k2_CoordAtt: C3k2 with Coordinate Attention")
    print("- C2f_CoordAtt: C2f with Coordinate Attention") 
    print("- EnhancedConv: Conv with optional Coordinate Attention")


# Auto-register when module is imported
register_custom_modules()