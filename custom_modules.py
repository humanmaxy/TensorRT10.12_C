import torch
import torch.nn as nn
from typing import Optional

try:
	from ultralytics.nn import modules as yolo_modules  # type: ignore
except Exception:  # pragma: no cover
	yolo_modules = None


class SEBlock(nn.Module):
	"""Squeeze-and-Excitation channel attention.

	Args:
		channels: number of input/output channels
		reduction: reduction ratio for squeeze bottleneck
	"""
	def __init__(self, channels: int, reduction: int = 16) -> None:
		super().__init__()
		bottleneck = max(1, channels // max(1, reduction))
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Conv2d(channels, bottleneck, kernel_size=1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(bottleneck, channels, kernel_size=1, bias=True),
			nn.Sigmoid(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
		w = self.avg_pool(x)
		w = self.fc(w)
		return x * w


class C3k2_SE(nn.Module):
	"""Wrapper around Ultralytics C3k2 with an appended SEBlock for channel attention.

	The constructor signature mirrors C3k2 to remain YAML-compatible.
	"""
	def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, e: float = 1.0, reduction: int = 16, **kwargs) -> None:
		super().__init__()
		if yolo_modules is None or not hasattr(yolo_modules, 'C3k2'):
			raise RuntimeError('Ultralytics modules not available or C3k2 not found')
		# Build the base C3k2 block
		self.c3 = yolo_modules.C3k2(c1, c2, n, shortcut, e, **kwargs)
		# Channel attention on the output channels
		self.se = SEBlock(c2, reduction=reduction)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		y = self.c3(x)
		return self.se(y)


def register_custom_modules() -> None:
	from ultralytics.nn import modules as _ym
	setattr(_ym, 'SEBlock', SEBlock)
	setattr(_ym, 'C3k2_SE', C3k2_SE)
	# Also inject into tasks module globals so parse_model `globals()[m]` can find them
	try:
		from ultralytics.nn import tasks as _yt
		setattr(_yt, 'SEBlock', SEBlock)
		setattr(_yt, 'C3k2_SE', C3k2_SE)
	except Exception:
		pass