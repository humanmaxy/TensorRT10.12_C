"""
Fixed advanced PyTorch modules for enhanced YOLO11 surface defect detection.
This version completely avoids the C3k2_Enhanced parameter issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, C3k2, C2f


class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation注意力机制
    Auto-detects input channels for YOLO compatibility
    """
    def __init__(self, reduction=16):
        super().__init__()
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_layers = None

    def _build_fc_layers(self, channels):
        """Build FC layers based on input channels."""
        return nn.Sequential(
            nn.Linear(channels, max(channels // self.reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // self.reduction, 8), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Build FC layers on first forward pass
        if self.fc_layers is None:
            self.fc_layers = self._build_fc_layers(c)
            self.fc_layers = self.fc_layers.to(x.device)
        
        y = self.avg_pool(x).view(b, c)
        y = self.fc_layers(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Auto-detects input channels for YOLO compatibility
    """
    def __init__(self, reduction=16, kernel_size=7):
        super().__init__()
        self.reduction = reduction
        
        # Channel attention components
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_layers = None
        
        # Spatial attention
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def _build_channel_layers(self, channels):
        """Build channel attention layers based on input channels."""
        return nn.ModuleDict({
            'fc1': nn.Conv2d(channels, max(channels // self.reduction, 8), 1, bias=False),
            'relu1': nn.ReLU(),
            'fc2': nn.Conv2d(max(channels // self.reduction, 8), channels, 1, bias=False),
        })

    def forward(self, x):
        # Build channel layers on first forward pass
        if self.channel_layers is None:
            self.channel_layers = self._build_channel_layers(x.size(1))
            self.channel_layers = self.channel_layers.to(x.device)
        
        # Channel attention
        avg_out = self.channel_layers['fc2'](self.channel_layers['relu1'](self.channel_layers['fc1'](self.avg_pool(x))))
        max_out = self.channel_layers['fc2'](self.channel_layers['relu1'](self.channel_layers['fc1'](self.max_pool(x))))
        channel_out = self.sigmoid(avg_out + max_out)
        x = x * channel_out
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_out
        
        return x


class ECA(nn.Module):
    """
    Efficient Channel Attention
    Auto-detects input channels for YOLO compatibility
    """
    def __init__(self, gamma=2, b=1):
        super().__init__()
        self.gamma = gamma
        self.b = b
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = None
        self.sigmoid = nn.Sigmoid()

    def _build_conv(self, channels):
        """Build 1D conv based on input channels."""
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + self.b) / self.gamma))
        k_size = max(t if t % 2 else t + 1, 3)  # Ensure minimum kernel size
        return nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        # Build conv on first forward pass
        if self.conv is None:
            self.conv = self._build_conv(x.size(1))
            self.conv = self.conv.to(x.device)
        
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SPP_Enhanced(nn.Module):
    """
    Enhanced Spatial Pyramid Pooling
    """
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class FPN_Enhanced(nn.Module):
    """
    Enhanced Feature Pyramid Network module
    """
    def __init__(self, channels):
        super().__init__()
        self.lateral_conv = nn.Conv2d(channels, channels, 1)
        self.fpn_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        lateral = self.lateral_conv(x)
        fpn_out = self.fpn_conv(lateral)
        return self.relu(fpn_out)


def register_advanced_modules_fixed():
    """
    Register fixed advanced modules with ultralytics framework.
    """
    import ultralytics.nn.tasks as tasks_module
    
    # Register the fixed modules
    tasks_module.SEAttention = SEAttention
    tasks_module.CBAM = CBAM
    tasks_module.ECA = ECA
    tasks_module.SPP_Enhanced = SPP_Enhanced
    tasks_module.FPN_Enhanced = FPN_Enhanced
    
    print("Fixed advanced modules registered successfully:")
    print("- SEAttention: Squeeze-and-Excitation注意力 (稳定版)")
    print("- CBAM: 卷积块注意力模块 (稳定版)")
    print("- ECA: 高效通道注意力 (稳定版)")
    print("- SPP_Enhanced: 增强空间金字塔池化")
    print("- FPN_Enhanced: 增强特征金字塔网络")


# Auto-register when module is imported
register_advanced_modules_fixed()