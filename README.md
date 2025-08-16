# Surface Defect Detection on YOLOv11 (with P2 head)

## Setup

```bash
python3 -m venv /workspace/.venv
/workspace/.venv/bin/pip install -U pip setuptools wheel
/workspace/.venv/bin/pip install ultralytics==8.3.70 torch torchvision opencv-python pyyaml onnx onnxsim
/workspace/.venv/bin/pip install streamlit pandas plotly
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

## Train (CLI)
```bash
/workspace/.venv/bin/python /workspace/train_surface_defect.py \
  --model /workspace/models/yolo11_surface_defect_p2.yaml \
  --data /workspace/data/surface_defect.yaml \
  --epochs 200 --batch 16 --imgsz 640 --device 0 \
  --project runs/train --name yolo11-surface-p2
```

## Inference (CLI)
```bash
/workspace/.venv/bin/python /workspace/infer_surface_defect.py \
  --weights runs/train/yolo11-surface-p2/weights/best.pt \
  --source /path/to/images_or_dir \
  --imgsz 640 --conf 0.25 --iou 0.6 --device 0 --save
```

## GUI (Streamlit)
- Configure model (scale、是否启用P2、C2PSA)、类别数、训练/验证路径、训练超参
- 一键启动训练并在页面显示训练曲线（precision/recall/mAP、损失），以及验证集预测预览图

```bash
/workspace/.venv/bin/streamlit run /workspace/app.py --server.address=0.0.0.0 --server.port=8501
```

打开浏览器访问: http://localhost:8501
