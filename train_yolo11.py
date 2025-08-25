import sys, os
import argparse
from typing import Optional

# Prefer absolute paths for reliability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
	from ultralytics import YOLO
except Exception as e:
	print('Failed to import ultralytics. Please `pip install ultralytics`:', e)
	raise

# Optional custom modules
sys.path.append(SCRIPT_DIR)
try:
	from custom_modules import register_custom_modules
	register_custom_modules()
	from custom_modules import C3k2_SE  # noqa: F401
except Exception as e:
	print('custom module registration failed:', e)
	pass


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='models/yolo11_surface_defect_p2.yaml')
	parser.add_argument('--data', type=str, default='zhengchang.yaml')
	parser.add_argument('--epochs', type=int, default=200)
	parser.add_argument('--batch', type=int, default=4)
	parser.add_argument('--imgsz', type=int, default=640)
	parser.add_argument('--device', type=str, default='0')
	parser.add_argument('--project', type=str, default='runs/train')
	parser.add_argument('--name', type=str, default='yolo11-surface-0823')
	parser.add_argument('--exist-ok', action='store_true', help='Allow existing project/name dir (recommended for resume)')
	parser.add_argument('--resume', action='store_true')
	parser.add_argument('--weights', type=str, default='', help='本地预训练权重路径(.pt)。为空表示不指定')
	parser.add_argument('--offline', action='store_true', help='离线模式，不从互联网下载任何权重（默认建议开启）')
	parser.add_argument('--save-period', type=int, default=0, help='Save checkpoint every N epochs (0 disables)')
	parser.add_argument('--workers', type=int, default=0, help='Number of dataloader workers')
	return parser.parse_args()


def to_abs_path(path_str: str) -> str:
	if os.path.isabs(path_str):
		return path_str
	return os.path.abspath(os.path.join(SCRIPT_DIR, path_str))


def ensure_dirs(project_dir: str, run_name: str) -> str:
	"""Create project/name/weights dirs and return the absolute run directory."""
	project_abs = to_abs_path(project_dir)
	run_dir = os.path.join(project_abs, run_name)
	weights_dir = os.path.join(run_dir, 'weights')
	os.makedirs(weights_dir, exist_ok=True)
	return run_dir


def resolve_resume_checkpoint(args, run_dir: str) -> Optional[str]:
	"""Resolve which checkpoint to resume from.
	Rule:
	1) If --weights is a valid .pt, use it
	2) Else use {project}/{name}/weights/best.pt if it exists
	3) Else return None (fresh start)
	"""
	if args.weights and os.path.isfile(to_abs_path(args.weights)):
		return to_abs_path(args.weights)

	best_in_named_run = os.path.join(run_dir, 'weights', 'best.pt')
	if os.path.isfile(best_in_named_run):
		return best_in_named_run

	return None


def build_hyp(args):
	# High-precision defaults for small defects
	# - reduce strong augmentations early; keep more resolution
	return dict(
		imgsz=args.imgsz,
		batch=args.batch,
		epochs=args.epochs,
		device=args.device,
		optimizer='SGD',
		cos_lr=True,
		close_mosaic=20,
		hsv_h=0.005,
		hsv_s=0.5,
		hsv_v=0.3,
		mosaic=0.9,
		mixup=0.1,
		copy_paste=0.3,
		flipud=0.0,
		fliplr=0.5,
		translate=0.05,
		scale=0.5,
		shear=0.0,
		perspective=0.0,
		box=10.0,
		cls=0.5,
		dfl=2.0,
		multi_scale=True,
		cache='disk',
		conf=0.01,
		iou=0.5,
		workers=args.workers,
		patience=300,
		save_period=args.save_period,
	)


def main():
	args = parse_args()

	# Resolve and create save directories early to avoid "No such file or directory" on saving weights
	run_dir = ensure_dirs(args.project, args.name)
	project_abs = to_abs_path(args.project)
	print(f'Project directory: {project_abs}')
	print(f'Run directory: {run_dir}')

	# Offline-safe: do not trigger auto-downloads. Only use local weights if provided.
	pretrained_flag = False

	if args.resume:
		ckpt_path = resolve_resume_checkpoint(args, run_dir)
		if ckpt_path and os.path.isfile(ckpt_path):
			print(f'Resuming from: {ckpt_path}')
			model = YOLO(ckpt_path)
			model.train(
				resume=True,
				project=project_abs,
				name=args.name,
				exist_ok=True,
				device=args.device,
				workers=args.workers,
				save_period=args.save_period,
			)
			return
		else:
			print('未找到 best.pt，改为重新开始训练。')

	# Fresh training path
	model = YOLO(to_abs_path(args.model))
	if args.weights:
		w_abs = to_abs_path(args.weights)
		if not os.path.isfile(w_abs):
			raise FileNotFoundError(f'指定的 --weights 不存在: {w_abs}')
		print(f'Loading custom pretrained weights: {w_abs}')
		model.load(w_abs)

	hyp = build_hyp(args)

	model.train(
		data=to_abs_path(args.data) if os.path.exists(to_abs_path(args.data)) else args.data,
		project=project_abs,
		name=args.name,
		pretrained=pretrained_flag,
		exist_ok=args.exist_ok,
		**hyp,
	)


if __name__ == '__main__':
	main()