
# Industrial YOLO11 部署指南

## 🚀 生产环境部署

### 1. 环境准备
```bash
# 创建虚拟环境
python -m venv industrial_yolo11_env
source industrial_yolo11_env/bin/activate  # Linux/Mac
# industrial_yolo11_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 模型优化
```python
# 模型量化
import torch
model = torch.load('best.pt')
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# ONNX导出
torch.onnx.export(
    model, 
    dummy_input,
    "industrial_yolo11.onnx",
    opset_version=11
)
```

### 3. 推理服务
```python
# Flask API服务示例
from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)
model = load_model('industrial_yolo11.onnx')

@app.route('/detect', methods=['POST'])
def detect_defects():
    # 接收X光图像
    image = request.files['image']
    
    # 预处理
    img_array = preprocess_xray_image(image)
    
    # 推理
    detections = model(img_array)
    
    # 后处理
    results = postprocess_detections(detections)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 4. 边缘设备部署
```bash
# TensorRT优化 (NVIDIA设备)
trtexec --onnx=industrial_yolo11.onnx \
        --saveEngine=industrial_yolo11.trt \
        --fp16

# OpenVINO优化 (Intel设备)  
mo --input_model industrial_yolo11.onnx \
   --output_dir openvino_model
```

## 📊 性能监控

### 关键指标
- **检测精度**: mAP@0.5, mAP@0.5:0.95
- **推理速度**: FPS, 延迟
- **内存使用**: GPU/CPU内存占用
- **微缺陷检出率**: 特别关注15微米级别目标

### 监控脚本
```python
import time
import psutil
import torch

def monitor_performance(model, test_data):
    metrics = {
        'fps': 0,
        'memory_usage': 0,
        'gpu_memory': 0,
        'accuracy': 0
    }
    
    # 性能测试
    start_time = time.time()
    with torch.no_grad():
        for data in test_data:
            output = model(data)
    end_time = time.time()
    
    metrics['fps'] = len(test_data) / (end_time - start_time)
    metrics['memory_usage'] = psutil.virtual_memory().percent
    
    if torch.cuda.is_available():
        metrics['gpu_memory'] = torch.cuda.memory_allocated() / 1024**3
        
    return metrics
```

## 🔧 故障排除

### 常见问题
1. **内存不足**: 减少batch_size或使用模型量化
2. **推理速度慢**: 使用TensorRT/OpenVINO优化
3. **检测精度低**: 检查数据质量和标注准确性
4. **微缺陷漏检**: 调整micro_conf_thres参数

### 性能调优
```python
# 推理优化配置
inference_config = {
    'conf_thres': 0.25,      # 主检测置信度
    'iou_thres': 0.45,       # NMS IoU阈值
    'micro_conf_thres': 0.15, # 微缺陷置信度
    'micro_iou_thres': 0.3,  # 微缺陷NMS阈值
    'max_det': 1000,         # 最大检测数
    'agnostic_nms': False    # 类别无关NMS
}
```

## 📈 持续优化

### 数据收集
- 收集更多真实X光焊缝图像
- 增加边缘案例样本
- 提高标注质量和一致性

### 模型改进
- 调整蛇形卷积参数
- 优化BiFPN层数和通道配置
- 改进微缺陷检测阈值

### 部署优化
- 模型剪枝和量化
- 推理引擎优化
- 硬件加速配置
