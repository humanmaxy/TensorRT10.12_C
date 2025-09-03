"""
Simple Modules for YOLO11
简单模块 - 提供标准的C3k2等模块
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv


class C3k2(nn.Module):
    """
    标准C3k2模块 - 简化版本，避免复杂参数
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Conv(c_, c_, 3) for _ in range(n)))
        self.add = shortcut and c1 == c2
        
    def forward(self, x):
        y = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
        return x + y if self.add else y


def register_simple_modules():
    """注册简单模块"""
    try:
        import ultralytics.nn.tasks as tasks
        import ultralytics.nn.modules as modules
        
        # 注册标准C3k2模块
        tasks.C3k2 = C3k2
        modules.C3k2 = C3k2
        
        print("✅ Simple modules registered: C3k2")
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to register simple modules: {e}")
        return False


# 自动注册
register_simple_modules()