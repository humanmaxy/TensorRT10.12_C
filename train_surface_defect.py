import os
import argparse
from ultralytics import YOLO


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='models/yolo11_surface_defect_p2.yaml')
	parser.add_argument('--data', type=str, default='data/surface_defect.yaml')
	parser.add_argument('--epochs', type=int, default=200)
	parser.add_argument('--batch', type=int, default=16)
	parser.add_argument('--imgsz', type=int, default=640)
	parser.add_argument('--device', type=str, default='0')
	parser.add_argument('--project', type=str, default='runs/train')
	parser.add_argument('--name', type=str, default='yolo11-surface-p2')
	parser.add_argument('--resume', action='store_true')
	return parser.parse_args()


def main():
	args = parse_args()
	# Create model
	model = YOLO(args.model)

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
		workers=8,
		patience=100,
	)

	if args.resume:
		model.train(resume=True)
	else:
		model.train(
			data=args.data,
			project=args.project,
			name=args.name,
			pretrained=True,
			**hyp,
		)


if __name__ == '__main__':
	main()