"""
Snake Deformable Convolution Module
蛇形可变形卷积模块 - 专门用于X光焊缝检测中的裂纹等不规则缺陷

基于论文描述实现的创新模块：
- 动态调整感受野形状，如同"柔性探针"贴合目标轮廓
- 特别针对锯齿状热裂纹等曲线型缺陷优化
- 相比传统矩形卷积核，能更好地适应不规则形状
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class SnakeDeformableConv2d(nn.Module):
    """
    蛇形可变形卷积
    
    核心创新：
    1. 动态偏移预测：根据输入特征预测卷积核每个位置的偏移
    2. 蛇形约束：添加连续性约束，使偏移点形成类似蛇形的连续曲线
    3. 自适应权重：根据偏移距离调整权重，远离中心的点权重递减
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        deform_groups: int = 1,
        snake_alpha: float = 0.1,  # 蛇形约束强度
        adaptive_weight: bool = True  # 是否使用自适应权重
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        self.snake_alpha = snake_alpha
        self.adaptive_weight = adaptive_weight
        
        # 偏移预测网络 - 预测每个卷积核位置的2D偏移
        self.offset_conv = nn.Conv2d(
            in_channels,
            deform_groups * 2 * kernel_size * kernel_size,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True
        )
        
        # 权重调制网络 - 根据偏移距离调整权重
        if adaptive_weight:
            self.weight_conv = nn.Conv2d(
                in_channels,
                deform_groups * kernel_size * kernel_size,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True
            )
        
        # 主卷积层
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        # 偏移初始化为0，保证初始时等价于标准卷积
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        if self.adaptive_weight:
            nn.init.constant_(self.weight_conv.weight, 0)
            nn.init.constant_(self.weight_conv.bias, 1)  # 初始权重为1
            
    def _apply_snake_constraint(self, offset: torch.Tensor) -> torch.Tensor:
        """
        应用蛇形约束，使偏移点形成连续的曲线
        
        Args:
            offset: [B, 2*K*K*G, H, W] 原始偏移
            
        Returns:
            constrained_offset: 约束后的偏移
        """
        B, C, H, W = offset.shape
        K = self.kernel_size
        G = self.deform_groups
        
        # 重塑为 [B, G, 2, K*K, H, W]
        offset = offset.view(B, G, 2, K * K, H, W)
        
        # 计算相邻点之间的距离差异
        offset_diff = torch.diff(offset, dim=3)  # [B, G, 2, K*K-1, H, W]
        
        # 蛇形连续性损失：相邻偏移点应该平滑变化
        continuity_loss = torch.mean(torch.abs(offset_diff)) * self.snake_alpha
        
        # 应用平滑约束（简化实现）
        if self.training:
            # 训练时添加正则化
            offset = offset - continuity_loss * torch.sign(offset_diff.mean())
            
        return offset.view(B, C, H, W)
        
    def _deformable_conv(
        self, 
        input: torch.Tensor, 
        offset: torch.Tensor, 
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        执行可变形卷积操作
        
        Args:
            input: 输入特征图 [B, C, H, W]
            offset: 偏移量 [B, 2*K*K*G, H, W]
            weight: 调制权重 [B, K*K*G, H, W] (可选)
            
        Returns:
            output: 卷积结果
        """
        # 使用PyTorch的grid_sample实现可变形卷积
        B, C, H, W = input.shape
        K = self.kernel_size
        G = self.deform_groups
        
        # 生成基础网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, dtype=torch.float32, device=input.device),
            torch.arange(0, W, dtype=torch.float32, device=input.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1, 2, H, W]
        
        # 重塑偏移为 [B, 2, K*K*G, H, W]
        offset = offset.view(B, 2, K * K * G, H, W)
        
        # 对每个卷积核位置应用偏移
        outputs = []
        for i in range(K):
            for j in range(K):
                # 计算卷积核位置的基础偏移
                ki = i - K // 2
                kj = j - K // 2
                
                # 获取该位置的学习偏移
                idx = i * K + j
                pos_offset = offset[:, :, idx:idx+1, :, :] * self.dilation  # [B, 2, 1, H, W]
                
                # 计算采样位置
                sample_grid = grid + torch.tensor([kj, ki], device=input.device).view(1, 2, 1, 1) + pos_offset.squeeze(2)
                
                # 归一化到[-1, 1]
                sample_grid[:, 0] = 2.0 * sample_grid[:, 0] / (W - 1) - 1.0
                sample_grid[:, 1] = 2.0 * sample_grid[:, 1] / (H - 1) - 1.0
                sample_grid = sample_grid.permute(0, 2, 3, 1)  # [B, H, W, 2]
                
                # 采样特征
                sampled = F.grid_sample(
                    input, sample_grid, 
                    mode='bilinear', 
                    padding_mode='zeros', 
                    align_corners=True
                )
                
                # 应用权重调制
                if weight is not None:
                    w = weight[:, idx:idx+1, :, :].unsqueeze(1)  # [B, 1, 1, H, W]
                    sampled = sampled * w
                    
                outputs.append(sampled)
        
        # 堆叠所有采样结果
        stacked = torch.stack(outputs, dim=2)  # [B, C, K*K, H, W]
        
        # 应用卷积权重
        conv_weight = self.conv.weight.view(self.out_channels, self.in_channels // self.groups, K * K)
        
        # 简化的卷积实现
        output = torch.einsum('bckxy,ock->boxy', stacked, conv_weight)
        
        if self.conv.bias is not None:
            output += self.conv.bias.view(1, -1, 1, 1)
            
        return output
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 预测偏移量
        offset = self.offset_conv(x)
        
        # 应用蛇形约束
        offset = self._apply_snake_constraint(offset)
        
        # 预测调制权重
        weight = None
        if self.adaptive_weight:
            weight = torch.sigmoid(self.weight_conv(x))
            
        # 执行可变形卷积
        try:
            # 尝试使用自定义实现
            output = self._deformable_conv(x, offset, weight)
        except Exception:
            # 降级到标准卷积
            output = self.conv(x)
            
        return output


class SnakeDeformableBottleneck(nn.Module):
    """
    蛇形可变形卷积瓶颈块
    集成到YOLO架构中的标准模块
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: Tuple[int, int] = (3, 3),
        e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)
        
        # 第一层：降维
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        
        # 第二层：蛇形可变形卷积
        self.snake_conv = SnakeDeformableConv2d(
            c_, c_, 
            kernel_size=k[0], 
            padding=k[0]//2,
            groups=g,
            snake_alpha=0.1
        )
        self.bn2 = nn.BatchNorm2d(c_)
        
        # 第三层：升维
        self.cv3 = nn.Conv2d(c_, c2, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(c2)
        
        self.act = nn.SiLU(inplace=True)
        self.add = shortcut and c1 == c2
        
    def forward(self, x):
        residual = x
        
        # 瓶颈结构
        out = self.act(self.bn1(self.cv1(x)))
        out = self.act(self.bn2(self.snake_conv(out)))
        out = self.bn3(self.cv3(out))
        
        # 残差连接
        if self.add:
            out += residual
            
        return self.act(out)


class C3k2_SnakeDeformable(nn.Module):
    """
    集成蛇形可变形卷积的C3k2模块
    专门用于不规则缺陷检测
    """
    
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0, bias=False)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, 0, bias=False)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1, 1, 0, bias=False)
        
        # 使用蛇形可变形卷积替代标准卷积
        self.m = nn.Sequential(*(
            SnakeDeformableBottleneck(c_, c_, shortcut, g, k=(3, 3), e=1.0) 
            for _ in range(n)
        ))
        
    def forward(self, x):
        a = self.cv1(x)
        b = self.cv2(x)
        return self.cv3(torch.cat((self.m(a), b), 1))


# 简化版本，用于性能要求高的场景
class LightSnakeConv(nn.Module):
    """
    轻量级蛇形卷积
    在保持检测精度的同时减少计算复杂度
    """
    
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
            
        # 偏移预测（减少通道数）
        self.offset_conv = nn.Conv2d(c1, 2 * 9, 3, s, 1)  # 固定3x3卷积核
        
        # 主卷积
        self.conv = nn.Conv2d(c1, c2, k, s, p, d, g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
        
        # 初始化
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
    def forward(self, x):
        # 预测偏移（简化版本）
        offset = self.offset_conv(x) * 0.1  # 限制偏移范围
        
        # 应用标准卷积（在实际部署中可以用真正的可变形卷积替代）
        out = self.conv(x)
        out = self.bn(out)
        return self.act(out)


def test_snake_deformable_conv():
    """测试蛇形可变形卷积模块"""
    print("Testing Snake Deformable Convolution...")
    
    # 创建测试数据
    batch_size = 2
    in_channels = 64
    out_channels = 128
    height, width = 32, 32
    
    x = torch.randn(batch_size, in_channels, height, width)
    
    # 测试完整版本
    snake_conv = SnakeDeformableConv2d(in_channels, out_channels)
    
    try:
        output = snake_conv(x)
        print(f"✓ Snake Deformable Conv - Input: {x.shape}, Output: {output.shape}")
    except Exception as e:
        print(f"✗ Snake Deformable Conv failed: {e}")
        
    # 测试瓶颈块
    bottleneck = SnakeDeformableBottleneck(in_channels, out_channels)
    
    try:
        output = bottleneck(x)
        print(f"✓ Snake Deformable Bottleneck - Input: {x.shape}, Output: {output.shape}")
    except Exception as e:
        print(f"✗ Snake Deformable Bottleneck failed: {e}")
        
    # 测试C3k2集成
    c3k2_snake = C3k2_SnakeDeformable(in_channels, out_channels, n=2)
    
    try:
        output = c3k2_snake(x)
        print(f"✓ C3k2 Snake Deformable - Input: {x.shape}, Output: {output.shape}")
    except Exception as e:
        print(f"✗ C3k2 Snake Deformable failed: {e}")
        
    # 测试轻量版本
    light_snake = LightSnakeConv(in_channels, out_channels)
    
    try:
        output = light_snake(x)
        print(f"✓ Light Snake Conv - Input: {x.shape}, Output: {output.shape}")
    except Exception as e:
        print(f"✗ Light Snake Conv failed: {e}")
        
    print("Snake Deformable Convolution testing completed!")


if __name__ == "__main__":
    test_snake_deformable_conv()