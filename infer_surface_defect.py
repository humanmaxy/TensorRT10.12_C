import os
import argparse
from ultralytics import YOLO
import glob
import cv2
import numpy as np
import torch
from torchvision.ops import nms


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', type=str, default='runs/train/yolo11-surface-p2/weights/best.pt')
	parser.add_argument('--source', type=str, default='assets/demo')
	parser.add_argument('--imgsz', type=int, default=640)
	parser.add_argument('--conf', type=float, default=0.01)
	parser.add_argument('--iou', type=float, default=0.5)
	parser.add_argument('--max_det', type=int, default=3000)
	parser.add_argument('--agnostic', action='store_true', help='Use class-agnostic NMS')
	parser.add_argument('--augment', action='store_true', help='Enable test-time augmentation (TTA)')
	parser.add_argument('--tile', action='store_true', help='Enable sliding-window tiling for large images')
	parser.add_argument('--tile_size', type=int, default=640, help='Tile size (pixels)')
	parser.add_argument('--overlap', type=int, default=128, help='Overlap between tiles (pixels)')
	parser.add_argument('--device', type=str, default='0')
	parser.add_argument('--save', action='store_true')
	parser.add_argument('--show', action='store_true')
	return parser.parse_args()


def _gen_positions(total: int, window: int, overlap: int):
	if total <= window:
		return [0]
	step = max(1, window - overlap)
	positions = []
	pos = 0
	while pos < total:
		pos = min(pos, total - window)
		if len(positions) == 0 or positions[-1] != pos:
			positions.append(pos)
		pos += step
	if positions[-1] != total - window:
		positions.append(total - window)
	return positions


def _tile_predict(model: YOLO, img_bgr: np.ndarray, args):
	h, w = img_bgr.shape[:2]
	xs = _gen_positions(w, args.tile_size, args.overlap)
	ys = _gen_positions(h, args.tile_size, args.overlap)
	all_boxes = []
	all_scores = []
	all_clses = []
	for y0 in ys:
		for x0 in xs:
			tile = img_bgr[y0:y0+args.tile_size, x0:x0+args.tile_size]
			res = model.predict(source=tile, imgsz=args.imgsz, conf=args.conf, iou=args.iou, max_det=args.max_det, agnostic_nms=args.agnostic, augment=args.augment, device=args.device, save=False, verbose=False)
			if not res:
				continue
			r = res[0]
			if not hasattr(r, 'boxes') or r.boxes is None or len(r.boxes) == 0:
				continue
			xyxy = r.boxes.xyxy.cpu().numpy()
			conf = r.boxes.conf.cpu().numpy()
			cls = r.boxes.cls.cpu().numpy()
			if xyxy.size == 0:
				continue
			xyxy[:, [0, 2]] += x0
			xyxy[:, [1, 3]] += y0
			all_boxes.append(xyxy)
			all_scores.append(conf)
			all_clses.append(cls)
	if not all_boxes:
		return np.zeros((0, 6), dtype=np.float32)
	boxes = torch.from_numpy(np.concatenate(all_boxes, axis=0)).float()
	scores = torch.from_numpy(np.concatenate(all_scores, axis=0).reshape(-1)).float()
	clses = torch.from_numpy(np.concatenate(all_clses, axis=0).reshape(-1)).float()
	keep_global = []
	if args.agnostic:
		keep_global = nms(boxes, scores, args.iou).cpu().numpy().tolist()
	else:
		for c in clses.unique().cpu().tolist():
			m = (clses == c)
			idxs = torch.where(m)[0]
			k = nms(boxes[idxs], scores[idxs], args.iou)
			keep_global.extend(idxs[k].cpu().numpy().tolist())
	keep_global = np.array(keep_global, dtype=np.int64)
	if keep_global.size == 0:
		return np.zeros((0, 6), dtype=np.float32)
	# Limit max_det
	if keep_global.size > args.max_det:
		order = scores[keep_global].argsort(descending=True)[:args.max_det]
		keep_global = keep_global[order.cpu().numpy()]
	final_boxes = boxes[keep_global].cpu().numpy()
	final_scores = scores[keep_global].cpu().numpy()
	final_clses = clses[keep_global].cpu().numpy()
	return np.concatenate([final_boxes, final_scores[:, None], final_clses[:, None]], axis=1)


def _save_vis(img_bgr: np.ndarray, dets: np.ndarray, out_path: str, class_names=None, thickness: int = 2):
	img = img_bgr.copy()
	for x1, y1, x2, y2, sc, cl in dets:
		p1 = (int(x1), int(y1))
		p2 = (int(x2), int(y2))
		cv2.rectangle(img, p1, p2, (0, 255, 0), thickness)
		label = f"{int(cl)}:{sc:.2f}"
		if class_names is not None and int(cl) < len(class_names):
			label = f"{class_names[int(cl)]}:{sc:.2f}"
		cv2.putText(img, label, (p1[0], max(0, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), max(1, thickness - 1), lineType=cv2.LINE_AA)
	cv2.imwrite(out_path, img)


def main():
	args = parse_args()
	model = YOLO(args.weights)
	if not args.tile:
		results = model.predict(source=args.source, imgsz=args.imgsz, conf=args.conf, iou=args.iou, max_det=args.max_det, agnostic_nms=args.agnostic, augment=args.augment, device=args.device, save=args.save, show=args.show)
		# Print brief summary
		for r in results:
			boxes = getattr(r, 'boxes', None)
			if boxes is not None:
				print(f"{getattr(r, 'path', 'image')}: {len(boxes)} detections")
		return
	# Tiled inference path
	src = args.source
	paths = []
	if os.path.isdir(src):
		exts = ('*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff')
		for e in exts:
			paths.extend(glob.glob(os.path.join(src, e)))
		paths = sorted(paths)
	else:
		paths = [src]
	if not paths:
		print('No images found for tiling')
		return
	out_dir = None
	if args.save:
		base = src if os.path.isdir(src) else os.path.dirname(src)
		out_dir = os.path.join(base, 'pred_tiled')
		os.makedirs(out_dir, exist_ok=True)
	names = getattr(model, 'names', None)
	for p in paths:
		img = cv2.imread(p)
		if img is None:
			print(f'Failed to read image: {p}')
			continue
		dets = _tile_predict(model, img, args)
		print(f"{p}: {len(dets)} detections (tiled)")
		if args.save and dets.size > 0:
			fname = os.path.splitext(os.path.basename(p))[0] + '_tiled.jpg'
			_save_vis(img, dets, os.path.join(out_dir, fname), class_names=names, thickness=2)


if __name__ == '__main__':
	main()