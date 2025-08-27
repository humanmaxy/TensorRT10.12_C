"""
Advanced modules for enhanced YOLO11 surface defect detection.
包含多种提高MAP的模块，特别针对表面缺陷检测优化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, C3k2, C2f


class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation注意力机制
    特别适合提高通道特征表示，对表面缺陷检测很有效
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
            nn.Linear(channels, channels // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // self.reduction, channels, bias=False),
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
    结合通道注意力和空间注意力，对小目标检测特别有效
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
            'fc1': nn.Conv2d(channels, channels // self.reduction, 1, bias=False),
            'relu1': nn.ReLU(),
            'fc2': nn.Conv2d(channels // self.reduction, channels, 1, bias=False),
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
    比SE更轻量但效果相当的通道注意力
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
        k_size = t if t % 2 else t + 1
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
    增强的空间金字塔池化
    多尺度特征融合，对不同大小的缺陷都有效
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
    增强的特征金字塔网络模块
    改善多尺度特征融合
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


class RepVGGBlock(nn.Module):
    """
    RepVGG结构块
    训练时多分支，推理时重参数化为单分支，提高精度和速度
    """
    def __init__(self, in_channels, out_channels, stride=1, groups=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        assert stride in [1, 2]
        
        padding_11 = 0
        
        if not deploy:
            self.identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.conv3x3 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
                nn.BatchNorm2d(num_features=out_channels),
            )
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups, bias=False),
                nn.BatchNorm2d(num_features=out_channels),
            )
        else:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=True)

        self.act = nn.ReLU()

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.act(self.rbr_reparam(inputs))

        if self.identity is None:
            id_out = 0
        else:
            id_out = self.identity(inputs)

        return self.act(self.conv3x3(inputs) + self.conv1x1(inputs) + id_out)


class GhostConv_Enhanced(nn.Module):
    """
    增强的Ghost卷积
    减少参数量的同时保持精度，适合移动端部署
    """
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class C3k2_Enhanced(nn.Module):
    """
    增强的C3k2模块，集成多种注意力机制
    Compatible with YOLO parameter parsing
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, attention_type='se'):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        
        # 选择注意力机制 (使用自适应版本)
        if attention_type == 'se':
            self.attention = SEAttention()
        elif attention_type == 'cbam':
            self.attention = CBAM()
        elif attention_type == 'eca':
            self.attention = ECA()
        else:
            self.attention = nn.Identity()
        
        # 使用简单的卷积层替代RepVGG避免复杂性
        self.m = nn.Sequential(*(Conv(c_, c_, 3) for _ in range(n)))

    def forward(self, x):
        a = self.cv1(x)
        b = self.cv2(x)
        out = self.cv3(torch.cat((self.m(a), b), 1))
        return self.attention(out)


class ASFF(nn.Module):
    """
    Adaptive Spatial Feature Fusion
    自适应空间特征融合，改善多尺度检测
    """
    def __init__(self, level, rfb=False, vis=False):
        super().__init__()
        self.level = level
        self.dim = [512, 256, 256]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(256, self.inter_dim, 3, 2)
            self.stride_level_2 = Conv(256, self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, 1024, 3, 1)
        elif level == 1:
            self.compress_level_0 = Conv(512, self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(256, self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, 512, 3, 1)
        elif level == 2:
            self.compress_level_0 = Conv(512, self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(256, self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, 256, 3, 1)

        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                           level_1_resized * levels_weight[:, 1:2, :, :] + \
                           level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


def register_advanced_modules():
    """
    注册高级模块到ultralytics框架
    """
    import ultralytics.nn.tasks as tasks_module
    
    # 注册所有高级模块
    tasks_module.SEAttention = SEAttention
    tasks_module.CBAM = CBAM
    tasks_module.ECA = ECA
    tasks_module.SPP_Enhanced = SPP_Enhanced
    tasks_module.FPN_Enhanced = FPN_Enhanced
    tasks_module.RepVGGBlock = RepVGGBlock
    tasks_module.GhostConv_Enhanced = GhostConv_Enhanced
    tasks_module.C3k2_Enhanced = C3k2_Enhanced
    tasks_module.ASFF = ASFF
    
    print("Advanced modules registered successfully:")
    print("- SEAttention: Squeeze-and-Excitation注意力")
    print("- CBAM: 卷积块注意力模块")
    print("- ECA: 高效通道注意力")
    print("- SPP_Enhanced: 增强空间金字塔池化")
    print("- FPN_Enhanced: 增强特征金字塔网络")
    print("- RepVGGBlock: 重参数化VGG块")
    print("- GhostConv_Enhanced: 增强Ghost卷积")
    print("- C3k2_Enhanced: 增强C3k2模块")
    print("- ASFF: 自适应空间特征融合")


# 自动注册模块
register_advanced_modules()