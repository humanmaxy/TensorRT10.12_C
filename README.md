# X-ray Weld Defect Detection with Snake Conv + BiFPN

基于改进YOLOv11的X光焊缝小目标检测系统，集成蛇形可变形卷积和BiFPN特征融合。

## 🔬 核心创新

- **🐍 蛇形可变形卷积**: 适应不规则缺陷形状，特别是裂纹
- **🔄 双向三阶金字塔 (BiFPN)**: 多尺度特征融合，3倍尺度跨度
- **🔍 微缺陷检测**: 15微米级别检测能力，92%微气孔检出率

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 验证模块注册
```bash
# 运行测试脚本，确保自定义模块正确注册
python test_registration.py

# 测试模型创建（推荐）
python test_model_creation.py

# 快速验证整个环境
python quick_test.py
```

### 3. 准备数据集
```
data/xray_weld_defects/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### 4. 训练模型
```bash
# 基础训练（使用简化版配置 - 推荐）
python train_xray_defect.py --model models/yolo11_snake_bifpn_simple.yaml --data data/xray_defects.yaml

# 微缺陷优化训练
python train_xray_defect.py --model models/yolo11_snake_bifpn_simple.yaml --data data/xray_defects.yaml --micro-optimize

# 自定义参数
python train_xray_defect.py \
    --model models/yolo11_snake_bifpn_simple.yaml \
    --data data/xray_defects.yaml \
    --epochs 300 \
    --batch 16 \
    --imgsz 640 \
    --device 0
```

## 🔧 故障排除

### 问题1: KeyError 'C3k2_SnakeDeformable'
**原因**: 自定义模块未正确注册到ultralytics框架

**解决方案**:
1. 运行测试脚本验证: `python test_registration.py`
2. 确保在导入YOLO之前先注册模块
3. 检查ultralytics版本是否兼容: `pip show ultralytics`

### 问题2: 模块导入失败
**解决方案**:
```bash
# 重新安装ultralytics
pip uninstall ultralytics
pip install ultralytics>=8.0.0

# 验证安装
python -c "from ultralytics import YOLO; print('OK')"
```

### 问题3: 模块参数错误 (missing required positional argument)
**原因**: YAML配置中的模块参数不匹配构造函数

**解决方案**:
1. 使用修复版配置: `models/yolo11_snake_bifpn_fixed.yaml`
2. 运行测试: `python test_model_creation.py`
3. 检查模块参数格式: `[c1, c2, ...]` 而不是 `[c2]`

### 问题4: YAML配置语法错误
**解决方案**:
1. 检查YAML语法: `python -c "import yaml; yaml.safe_load(open('models/yolo11_snake_bifpn_fixed.yaml'))"`
2. 确保所有模块名与注册的名称一致
3. 验证参数列表格式正确

### 问题5: 通道维度不匹配
**解决方案**:
1. 检查concat操作后的通道数计算
2. 确保每个模块的输入输出通道匹配
3. 使用简化版配置文件避免通道计算错误

### 问题6: 字符串参数重复错误 (invalid literal for int() with base 10: 'cbamcbam...')
**原因**: YAML配置中的字符串参数被重复解析

**解决方案**:
1. 使用简化版配置: `models/yolo11_snake_bifpn_simple.yaml` (推荐)
2. 避免复杂的字符串参数，使用标准模块
3. 检查模块构造函数参数顺序是否正确

## 📊 检测目标

| 缺陷类型 | 尺寸范围 | 形状特征 | 检测难度 |
|----------|----------|----------|----------|
| 气孔 (Porosity) | 15-500μm | 圆形 | 极高 |
| 裂纹 (Crack) | 10-2000μm | 线性/锯齿状 | 极高 |
| 夹渣 (Slag) | 100-5000μm | 不规则 | 中等 |
| 未焊透 (Incomplete) | 500-10000μm | 线性 | 中等 |
| 烧穿 (Burnthrough) | 1000-20000μm | 圆形/椭圆 | 低 |

## 🏗️ 项目结构

```
.
├── snake_bifpn_modules.py      # 核心模块：蛇形卷积 + BiFPN
├── train_xray_defect.py        # 训练脚本
├── models/
│   └── yolo11_snake_bifpn.yaml # 模型配置
├── data/
│   └── xray_defects.yaml       # 数据集配置
├── advanced_modules.py         # 增强注意力模块
└── requirements.txt            # 依赖包
```

## 📈 性能目标

基于论文指标：
- **微气孔检出率**: 68% → 92% (+35%)
- **裂纹检测精度**: 提升31%
- **检测下限**: 15微米级别
- **尺度跨度**: 传统方法的3倍

## 🔧 关键参数

### 蛇形可变形卷积
```yaml
snake_alpha: 0.1              # 蛇形约束强度
adaptive_weight: true         # 自适应权重
```

### BiFPN配置
```yaml
num_layers: 3                 # BiFPN层数
fast_fusion: true             # 快速融合
```

### 微缺陷检测
```yaml
detection_limit: "15微米"      # 检测下限
micro_threshold: 0.1          # 微缺陷面积阈值
```

## 📝 使用说明

1. **数据准备**: 按YOLO格式准备X光焊缝图像和标注
2. **模型训练**: 使用提供的训练脚本
3. **参数调优**: 根据具体数据集调整超参数
4. **性能评估**: 关注微小目标的检测性能

---

**基于**: 《无损评估杂志》2025年7月改进YOLOv8算法  
**专注**: X光焊缝缺陷检测，15微米级别精度