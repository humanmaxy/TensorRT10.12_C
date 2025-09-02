
# Industrial YOLO11 使用示例

## 1. 快速开始

```python
from yolo11_industrial_detector import IndustrialYOLO11Factory

# 创建模型
model = IndustrialYOLO11Factory.create_model('s', num_classes=5)

# 推理
import torch
input_image = torch.randn(1, 3, 640, 640)
detections = model(input_image)
```

## 2. 训练自定义数据集

```bash
# 准备数据集（YOLO格式）
mkdir -p data/xray_defects/{images,labels}/{train,val}

# 开始训练
python train_industrial_yolo11.py \
    --data data/xray_defects.yaml \
    --scale s \
    --epochs 300 \
    --batch 16
```

## 3. 微缺陷专用检测

```python
# 启用微缺陷检测模式
model = IndustrialYOLO11Factory.create_micro_optimized_model(num_classes=5)

# 获取微缺陷检测结果
main_output, micro_output = model(input_image, return_micro=True)
micro_cls, micro_reg, micro_conf = micro_output
```

## 4. 模块单独使用

```python
# 使用蛇形可变形卷积
from snake_deformable_conv import SnakeDeformableConv2d
snake_conv = SnakeDeformableConv2d(64, 128, kernel_size=3)

# 使用BiFPN
from bifpn_module import TripleBiFPN
bifpn = TripleBiFPN(channels=256, num_levels=3, num_layers=2)

# 使用微缺陷检测头
from micro_defect_head import MicroDefectHead
micro_head = MicroDefectHead([128, 256, 512], num_classes=5)
```

## 5. 性能优化配置

```python
# 速度优化
speed_model = IndustrialYOLO11Factory.create_speed_optimized_model()

# 精度优化  
precision_model = IndustrialYOLO11Factory.create_micro_optimized_model()
```
