import os
import argparse
from ultralytics import YOLO

# Import custom modules for enhanced attention mechanisms
import custom_modules  # This automatically registers our custom modules


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='models/yolo11_surface_defect_p2_coordatt_final.yaml')
	parser.add_argument('--data', type=str, default='data/surface_defect.yaml')
	parser.add_argument('--epochs', type=int, default=200)
	parser.add_argument('--batch', type=int, default=4)
	parser.add_argument('--imgsz', type=int, default=640)
	parser.add_argument('--device', type=str, default='0')
	parser.add_argument('--project', type=str, default='runs/train')
	parser.add_argument('--name', type=str, default='yolo11-surface-p2-coordatt')
	parser.add_argument('--resume', action='store_true')
	parser.add_argument('--weights', type=str, default='', help='本地预训练权重路径(.pt)。为空表示不指定')
	parser.add_argument('--offline', action='store_true', help='离线模式，不从互联网下载任何权重（默认建议开启）')
	return parser.parse_args()


def main():
	args = parse_args()
	
	print("🚀 Enhanced YOLO11 Training with Coordinate Attention")
	print(f"Model: {args.model}")
	print("Custom modules loaded: CoordAtt, C3k2_CoordAtt, C2f_CoordAtt, EnhancedConv")
	
	# Create model (always from YAML to avoid online downloads)
	model = YOLO(args.model)
	# Load custom pretrained weights if provided
	if args.weights:
		model.load(args.weights)

	# High-precision defaults for small defects
	# - reduce strong augmentations early; keep more resolution
	hyp = dict(
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
		mosaic=0.8,
		mixup=0.1,
		flipud=0.0,
		fliplr=0.5,
		translate=0.05,
		scale=0.4,
		shear=0.0,
		perspective=0.0,
		box=7.5,
		cls=0.7,
		dfl=1.5,
		workers=0,
		patience=100,
	)

	# Offline-safe: do not trigger auto-downloads. Only use local weights if provided.
	pretrained_flag = False

	if args.resume:
		model.train(resume=True)
	else:
		model.train(
			data=args.data,
			project=args.project,
			name=args.name,
			pretrained=pretrained_flag,
			**hyp,
		)


if __name__ == '__main__':
	main()