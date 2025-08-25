import torch
import torch.nn as nn
import torch.nn.functional as F
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


class _ConvBNAct(nn.Module):
	"""Lightweight Conv-BN-Activation fallback if Ultralytics Conv is unavailable."""

	def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None, g: int = 1, act: bool = True) -> None:
		super().__init__()
		if p is None:
			if isinstance(k, int):
				p = k // 2
			elif isinstance(k, tuple):
				p = (k[0] // 2, k[1] // 2)  # type: ignore
			else:
				p = 0
		self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, groups=g, bias=False)
		self.bn = nn.BatchNorm2d(c2)
		self.act = nn.SiLU(inplace=True) if act else nn.Identity()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.act(self.bn(self.conv(x)))


def _get_conv_impl():
	"""Return Ultralytics Conv if available, otherwise fallback."""
	if yolo_modules is not None and hasattr(yolo_modules, 'Conv'):
		return yolo_modules.Conv  # type: ignore[attr-defined]
	return _ConvBNAct


class RFD(nn.Module):
	"""Robust Feature Downsampling (RFD).

	Dual-branch downsampling to preserve fine textures during stride-2 reduction.
	- Branch A: kxk stride-s conv
	- Branch B: AvgPool stride-s + 1x1 conv
	Outputs are concatenated then fused by 1x1 conv to target channels.
	"""

	def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, act: bool = True) -> None:
		super().__init__()
		Conv = _get_conv_impl()
		c2a = c2 // 2
		c2b = c2 - c2a
		self.branch_a = Conv(c1, c2a, k, s, None, 1, act)
		self.pool = nn.AvgPool2d(kernel_size=s, stride=s, ceil_mode=False)
		self.branch_b = Conv(c1, c2b, 1, 1, 0, 1, act)
		self.fuse = Conv(c2, c2, 1, 1, 0, 1, act)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		a = self.branch_a(x)
		b = self.pool(x)
		b = self.branch_b(b)
		out = torch.cat([a, b], dim=1)
		return self.fuse(out)


class SPDConv(nn.Module):
	"""Spatially Preferential Dynamic Convolution.

	Combines anisotropic branches (k×1, 1×k) and 3×3 with learned soft weights.
	The stride can be 1 or 2, enabling it to replace downsampling convs.
	"""

	def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, reduction: int = 16, act: bool = True) -> None:
		super().__init__()
		Conv = _get_conv_impl()
		k = max(3, int(k))
		self.b1 = Conv(c1, c2, (k, 1), s, None, 1, act)
		self.b2 = Conv(c1, c2, (1, k), s, None, 1, act)
		self.b3 = Conv(c1, c2, 3, s, None, 1, act)
		# Gating network
		self.pool = nn.AdaptiveAvgPool2d(1)
		g_hidden = max(1, c1 // max(1, reduction))
		self.gate = nn.Sequential(
			nn.Conv2d(c1, g_hidden, 1, 1, 0, bias=True),
			nn.SiLU(inplace=True),
			nn.Conv2d(g_hidden, 3, 1, 1, 0, bias=True),
			nn.Softmax(dim=1),
		)
		self.fuse = Conv(c2, c2, 1, 1, 0, 1, act)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		w = self.gate(self.pool(x))  # B×3×1×1
		y = self.b1(x) * w[:, 0:1] + self.b2(x) * w[:, 1:2] + self.b3(x) * w[:, 2:3]
		return self.fuse(y)


class HAttention(nn.Module):
	"""Hybrid Channel-Spatial Attention (CBAM-like) with YAML-friendly signature.

	Accepts (c1, c2, ...) like Ultralytics modules, but only uses channel count.
	"""

	def __init__(self, c1: Optional[int] = None, c2: Optional[int] = None, reduction: int = 16, spatial_kernel: int = 5) -> None:
		super().__init__()
		channels = c2 if c2 is not None else c1
		self._channels: Optional[int] = channels
		self._reduction = reduction
		self._spk = spatial_kernel if spatial_kernel % 2 == 1 else spatial_kernel + 1
		self.ca_mlp: Optional[nn.Sequential] = None
		self.sa: Optional[nn.Conv2d] = None
		if channels is not None:
			self._build(channels)

	def _build(self, c: int) -> None:
		mid = max(1, c // max(1, self._reduction))
		self.ca_mlp = nn.Sequential(
			nn.Conv2d(c, mid, 1, bias=True),
			nn.SiLU(inplace=True),
			nn.Conv2d(mid, c, 1, bias=True),
		)
		self.sa = nn.Conv2d(2, 1, kernel_size=self._spk, padding=self._spk // 2, bias=True)
		self._channels = c

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self.ca_mlp is None or self.sa is None or self._channels != x.shape[1]:
			self._build(x.shape[1])
		# Channel attention with avg and max pooling
		avg = F.adaptive_avg_pool2d(x, 1)
		mx = F.adaptive_max_pool2d(x, 1)
		ca = torch.sigmoid(self.ca_mlp(avg) + self.ca_mlp(mx))  # type: ignore[arg-type]
		x = x * ca
		# Spatial attention
		avg_c = torch.mean(x, dim=1, keepdim=True)
		max_c, _ = torch.max(x, dim=1, keepdim=True)
		sa = torch.sigmoid(self.sa(torch.cat([avg_c, max_c], dim=1)))  # type: ignore[arg-type]
		return x * sa


class DWRSeg(nn.Module):
	"""Dilated Wider Residual Segment block.

	1×1 reduce -> parallel depthwise dilated convs (rates 1, 2, 3) -> 1×1 expand + residual.
	This increases receptive field for long, thin targets with minimal parameter growth.
	"""

	def __init__(self, c1: int, c2: Optional[int] = None, dilation_rates: tuple = (1, 2, 3), e: float = 0.5, act: bool = True) -> None:
		super().__init__()
		Conv = _get_conv_impl()
		c2 = c1 if c2 is None else c2
		mid = max(8, int(c2 * e))
		self.reduce = Conv(c1, mid, 1, 1, 0, 1, act)
		branches = []
		for d in dilation_rates:
			branches.append(nn.Sequential(
				nn.Conv2d(mid, mid, kernel_size=3, stride=1, padding=d, dilation=d, groups=mid, bias=False),
				nn.BatchNorm2d(mid),
				nn.SiLU(inplace=True) if act else nn.Identity(),
			))
		self.branches = nn.ModuleList(branches)
		self.project = Conv(mid * len(dilation_rates), c2, 1, 1, 0, 1, act)
		self.use_res = (c1 == c2)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		y = self.reduce(x)
		outs = [b(y) for b in self.branches]
		z = torch.cat(outs, dim=1)
		z = self.project(z)
		return z + x if self.use_res else z


def enable_eiou_loss() -> None:
	"""Best-effort monkey patch to enable EIoU loss inside Ultralytics if available.

	If the internal APIs change, this function quietly does nothing.
	"""
	try:
		from ultralytics.utils import loss as yolo_loss  # type: ignore

		def _bbox_iou_eiou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
			# box format: (x1,y1,x2,y2)
			x1 = torch.max(box1[:, None, 0], box2[:, 0])
			y1 = torch.max(box1[:, None, 1], box2[:, 1])
			x2 = torch.min(box1[:, None, 2], box2[:, 2])
			y2 = torch.min(box1[:, None, 3], box2[:, 3])
			inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
			area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
			area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)
			iou = inter / (area1[:, None] + area2 - inter + eps)
			# enclosing box
			cx1 = torch.min(box1[:, None, 0], box2[:, 0])
			cy1 = torch.min(box1[:, None, 1], box2[:, 1])
			cx2 = torch.max(box1[:, None, 2], box2[:, 2])
			cy2 = torch.max(box1[:, None, 3], box2[:, 3])
			cw = (cx2 - cx1).clamp(eps)
			ch = (cy2 - cy1).clamp(eps)
			# centers and sizes
			b1c_x = (box1[:, 0] + box1[:, 2]) / 2
			b1c_y = (box1[:, 1] + box1[:, 3]) / 2
			b2c_x = (box2[:, 0] + box2[:, 2]) / 2
			b2c_y = (box2[:, 1] + box2[:, 3]) / 2
			center_dist = (b1c_x[:, None] - b2c_x) ** 2 + (b1c_y[:, None] - b2c_y) ** 2
			# width/height mismatch
			w1 = (box1[:, 2] - box1[:, 0]).clamp(eps)
			h1 = (box1[:, 3] - box1[:, 1]).clamp(eps)
			w2 = (box2[:, 2] - box2[:, 0]).clamp(eps)
			h2 = (box2[:, 3] - box2[:, 1]).clamp(eps)
			w_diff2 = (w1[:, None] - w2) ** 2
			h_diff2 = (h1[:, None] - h2) ** 2
			eiou = iou - center_dist / (cw ** 2 + ch ** 2 + eps) - w_diff2 / (cw ** 2 + eps) - h_diff2 / (ch ** 2 + eps)
			return eiou.clamp(min=-1.0, max=1.0)

		# Try patching bbox_iou if present
		if hasattr(yolo_loss, 'bbox_iou'):
			yolo_loss.bbox_iou = _bbox_iou_eiou  # type: ignore
			return
	except Exception:
		# Silently ignore if internal API differs
		return


def register_custom_modules() -> None:
	from ultralytics.nn import modules as _ym
	setattr(_ym, 'SEBlock', SEBlock)
	setattr(_ym, 'C3k2_SE', C3k2_SE)
	setattr(_ym, 'RFD', RFD)
	setattr(_ym, 'SPDConv', SPDConv)
	setattr(_ym, 'HAttention', HAttention)
	setattr(_ym, 'DWRSeg', DWRSeg)
	# Also inject into tasks module globals so parse_model `globals()[m]` can find them
	try:
		from ultralytics.nn import tasks as _yt
		setattr(_yt, 'SEBlock', SEBlock)
		setattr(_yt, 'C3k2_SE', C3k2_SE)
		setattr(_yt, 'RFD', RFD)
		setattr(_yt, 'SPDConv', SPDConv)
		setattr(_yt, 'HAttention', HAttention)
		setattr(_yt, 'DWRSeg', DWRSeg)
	except Exception:
		pass