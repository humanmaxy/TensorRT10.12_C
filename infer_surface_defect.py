import os
import argparse
from ultralytics import YOLO

# Import custom modules for enhanced attention mechanisms
import custom_modules  # This automatically registers our custom modules


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', type=str, default='runs/train/yolo11-surface-p2-coordatt/weights/best.pt')
	parser.add_argument('--source', type=str, default='assets/demo')
	parser.add_argument('--imgsz', type=int, default=640)
	parser.add_argument('--conf', type=float, default=0.25)
	parser.add_argument('--iou', type=float, default=0.6)
	parser.add_argument('--device', type=str, default='0')
	parser.add_argument('--save', action='store_true')
	parser.add_argument('--show', action='store_true')
	return parser.parse_args()


def main():
	args = parse_args()
	
	print("🔍 Enhanced YOLO11 Inference with Coordinate Attention")
	print(f"Model weights: {args.weights}")
	print("Custom modules loaded: CoordAtt, C3k2_CoordAtt, C2f_CoordAtt, EnhancedConv")
	
	model = YOLO(args.weights)
	results = model.predict(source=args.source, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, save=args.save, show=args.show)
	# Print brief summary
	for r in results:
		boxes = getattr(r, 'boxes', None)
		if boxes is not None:
			print(f"{getattr(r, 'path', 'image')}: {len(boxes)} detections")


if __name__ == '__main__':
	main()