# Enhanced YOLO11 with Coordinate Attention (CoordAtt)

This project integrates Coordinate Attention mechanisms into YOLO11 for improved surface defect detection, specifically designed to boost MAP (Mean Average Precision) values for small defect detection.

## 🚀 Features Added

### CoordAtt Module
- **Coordinate Attention mechanism** that captures position-sensitive information along both horizontal and vertical directions
- **Spatial feature enhancement** particularly beneficial for surface defect detection where spatial relationships are crucial
- **Efficient design** with configurable reduction ratio to balance performance and computational cost

### Enhanced Architecture
- Strategic placement of CoordAtt modules in the detection head
- Focused attention on P2, P3, and P4 feature maps for multi-scale defect detection
- Preserves the efficient YOLO11 backbone while enhancing feature representation

## 📁 New Files Added

1. **`custom_modules.py`** - Contains the CoordAtt implementation and enhanced modules
2. **`models/yolo11_surface_defect_p2_coordatt_final.yaml`** - Enhanced model configuration
3. **`test_coordatt_integration.py`** - Comprehensive integration tests
4. **`test_simple_coordatt.py`** - Basic functionality tests

## 🔧 Updated Files

1. **`train_surface_defect.py`** - Now imports custom modules and uses enhanced model by default
2. **`infer_surface_defect.py`** - Updated to support custom modules
3. **`app.py`** - Added CoordAtt toggle in Streamlit interface

## 🧪 Usage

### Training with CoordAtt
```bash
# Using the enhanced model
/workspace/.venv/bin/python train_surface_defect.py \
  --model models/yolo11_surface_defect_p2_coordatt_final.yaml \
  --data data/surface_defect.yaml \
  --epochs 200 --batch 16 --imgsz 640 --device 0 \
  --project runs/train --name yolo11-surface-p2-coordatt
```

### Inference with Enhanced Model
```bash
/workspace/.venv/bin/python infer_surface_defect.py \
  --weights runs/train/yolo11-surface-p2-coordatt/weights/best.pt \
  --source /path/to/images \
  --imgsz 640 --conf 0.25 --iou 0.6 --device 0 --save
```

### GUI Training (Streamlit)
```bash
/workspace/.venv/bin/streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```
- Now includes "启用Coordinate Attention (提升MAP)" checkbox option

## 🔬 Technical Details

### CoordAtt Module Implementation
```python
class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super().__init__()
        # Coordinate encoding layers
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))    # [B,C,H,1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))    # [B,C,1,W]
        
        # Shared convolution layer
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        # Direction attention convolutions  
        self.conv_h = nn.Conv2d(mip, inp, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, inp, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate attention weights for both directions
        # Apply spatial weighting to enhance important features
        return identity * att_h * att_w
```

### Key Benefits for Surface Defect Detection
1. **Improved small defect detection** through enhanced spatial attention
2. **Better feature localization** with coordinate-aware processing
3. **Minimal computational overhead** with efficient attention design
4. **Preserved model compatibility** with existing YOLO11 architecture

## 📊 Expected Improvements

- **Enhanced MAP values** particularly for small surface defects
- **Better spatial feature representation** leading to more accurate localization
- **Improved detection of elongated defects** (scratches, cracks) through directional attention
- **Maintained inference speed** with minimal computational overhead

## 🧪 Testing

Run the integration tests to verify everything works:
```bash
/workspace/.venv/bin/python test_coordatt_integration.py
/workspace/.venv/bin/python test_simple_coordatt.py
```

## 🎯 Integration Strategy

The CoordAtt modules are strategically placed at:
- **P2 head (128 channels)** - Critical for tiny defect detection
- **P3 head (256 channels)** - Important for small defect features  
- **P4 head (512 channels)** - Enhanced medium-scale feature processing

This placement maximizes the benefit for multi-scale surface defect detection while maintaining computational efficiency.

## 💡 Future Enhancements

- Adaptive reduction ratios based on feature map size
- Channel attention combined with spatial attention
- Multi-head attention mechanisms for complex defect patterns
- Dynamic attention weighting based on defect type