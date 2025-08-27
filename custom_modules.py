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
    
    The module automatically adapts to input channel dimensions.
    """
    
    def __init__(self, reduction=32):
        super().__init__()
        self.reduction = max(reduction, 1)
        
        # Pooling layers
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.sigmoid = nn.Sigmoid()
        
        # Placeholder for lazy initialization
        self.conv_layers = None

    def _build_conv_layers(self, inp_channels):
        """Build convolution layers based on input channels."""
        mip = max(8, inp_channels // self.reduction)
        
        conv_layers = nn.ModuleDict({
            'conv1': nn.Conv2d(inp_channels, mip, 1, bias=False),
            'bn1': nn.BatchNorm2d(mip),
            'act': nn.Hardswish(),
            'conv_h': nn.Conv2d(mip, inp_channels, 1, bias=False),
            'conv_w': nn.Conv2d(mip, inp_channels, 1, bias=False),
        })
        
        return conv_layers

    def forward(self, x):
        identity = x
        n, c, h, w = x.shape
        
        # Build conv layers on first forward pass if needed
        if self.conv_layers is None:
            self.conv_layers = self._build_conv_layers(c)
            # Move to same device as input
            self.conv_layers = self.conv_layers.to(x.device)
        
        # Coordinate encoding
        x_h = self.pool_h(x)  # [B,C,H,1]
        x_w = self.pool_w(x)  # [B,C,1,W]
        
        # Process both directions
        y_h = self.conv_layers['act'](self.conv_layers['bn1'](self.conv_layers['conv1'](x_h)))
        y_w = self.conv_layers['act'](self.conv_layers['bn1'](self.conv_layers['conv1'](x_w)))
        
        # Generate attention weights
        att_h = self.sigmoid(self.conv_layers['conv_h'](y_h))  # [B,C,H,1]
        att_w = self.sigmoid(self.conv_layers['conv_w'](y_w))  # [B,C,1,W]
        
        # Apply spatial weighting
        return identity * att_h * att_w


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