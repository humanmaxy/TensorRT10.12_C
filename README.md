# Surface Defect Detection on YOLOv11 (with P2 head)

## Setup

```bash
python3 -m venv /workspace/.venv
/workspace/.venv/bin/pip install -U pip setuptools wheel
/workspace/.venv/bin/pip install ultralytics==8.3.70 torch torchvision opencv-python pyyaml onnx onnxsim
```

## Data
- Edit `data/surface_defect.yaml` to match your dataset root and class names.
- Dataset should be in YOLO format:
```
<root>/
  images/
    train/*.jpg
    val/*.jpg
  labels/
    train/*.txt
    val/*.txt
```

## Train
```bash
/workspace/.venv/bin/python /workspace/train_surface_defect.py \
  --model /workspace/models/yolo11_surface_defect_p2.yaml \
  --data /workspace/data/surface_defect.yaml \
  --epochs 200 --batch 16 --imgsz 640 --device 0 \
  --project runs/train --name yolo11-surface-p2
```

## Inference
```bash
/workspace/.venv/bin/python /workspace/infer_surface_defect.py \
  --weights runs/train/yolo11-surface-p2/weights/best.pt \
  --source /path/to/images_or_dir \
  --imgsz 640 --conf 0.25 --iou 0.6 --device 0 --save
```
