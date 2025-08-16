import os
import argparse
from ultralytics import YOLO


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='models/yolo11_surface_defect_p2.yaml')
	parser.add_argument('--data', type=str, default='data/surface_defect.yaml')
	parser.add_argument('--epochs', type=int, default=200)
	parser.add_argument('--batch', type=int, default=4)
	parser.add_argument('--imgsz', type=int, default=640)
	parser.add_argument('--device', type=str, default='0')
	parser.add_argument('--project', type=str, default='runs/train')
	parser.add_argument('--name', type=str, default='yolo11-surface-p2')
	parser.add_argument('--resume', action='store_true')
	parser.add_argument('--weights', type=str, default='', help='预训练权重路径(.pt)。为空表示不指定')
	parser.add_argument('--pretrained', action='store_true', help='使用内置预训练权重（与当前模型匹配）')
	return parser.parse_args()


def main():
	args = parse_args()
	# Create model
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

	# If custom weights are provided, do not also auto-load built-in pretrained
	use_pretrained = (args.pretrained and not args.weights)

	if args.resume:
		model.train(resume=True)
	else:
		model.train(
			data=args.data,
			project=args.project,
			name=args.name,
			pretrained=use_pretrained,
			**hyp,
		)


if __name__ == '__main__':
	main()