# 🔧 故障排除指南

## 常见错误及解决方案

### ❌ 错误1: KeyError 'C3k2_SnakeDeformable'
**症状**: 模块未找到错误
**原因**: 自定义模块未正确注册

**解决步骤**:
```bash
# 1. 验证模块注册
python quick_verify.py

# 2. 如果失败，运行详细调试
python debug_model.py

# 3. 检查内置模块
python check_builtin_modules.py
```

### ❌ 错误2: invalid literal for int() with base 10: 'cbamcbam...'
**症状**: 字符串参数重复错误
**原因**: YAML配置中的字符串参数被重复解析

**解决方案**:
1. 使用简化版配置: `models/yolo11_snake_bifpn_simple.yaml`
2. 避免复杂的字符串参数
3. 使用标准模块替代增强模块

### ❌ 错误3: 模型创建失败但无错误信息
**症状**: 显示"Failed to create model:"但没有具体错误
**原因**: 错误被捕获但未正确显示

**解决步骤**:
```bash
# 运行详细调试脚本
python debug_model.py
```

### ❌ 错误4: 模块参数不匹配
**症状**: missing required positional argument
**原因**: YAML中的参数与模块构造函数不匹配

**解决方案**:
1. 检查模块构造函数: `def __init__(self, c1, c2, ...)`
2. 确保YAML参数格式: `[c1, c2, ...]`
3. 使用简化版配置避免复杂参数

## 🛠️ 调试工具使用指南

### 1. 快速验证 (推荐首选)
```bash
python quick_verify.py
```
- 快速检查基本设置
- 测试简化版配置
- 提供简单的错误分析

### 2. 详细调试
```bash
python debug_model.py
```
- 逐步调试模型创建过程
- 显示完整的错误traceback
- 检查YAML语法和模块可用性
- 提供最小化配置作为fallback

### 3. 检查内置模块
```bash
python check_builtin_modules.py
```
- 查看ultralytics内置的模块
- 避免重复定义已有模块
- 了解可用的标准模块

### 4. 模块注册测试
```bash
python test_registration.py
```
- 测试模块注册机制
- 验证所有自定义模块是否可用
- 检查注册的完整性

## 🎯 推荐的调试流程

### 步骤1: 环境检查
```bash
# 检查Python和依赖
python --version
pip list | grep ultralytics
pip list | grep torch
```

### 步骤2: 快速验证
```bash
python quick_verify.py
```

### 步骤3: 如果失败，详细调试
```bash
python debug_model.py
```

### 步骤4: 检查内置模块
```bash
python check_builtin_modules.py
```

### 步骤5: 尝试安全配置
```bash
# 如果自定义模块有问题，先用安全配置测试
python train_xray_defect.py --model models/yolo11_safe.yaml --data data/xray_defects.yaml
```

## 🔄 配置文件优先级

### 推荐使用顺序:
1. **`yolo11_snake_bifpn_simple.yaml`** - 简化版，推荐
2. **`yolo11_safe.yaml`** - 安全版，只用内置模块
3. **`yolo11_snake_bifpn_fixed.yaml`** - 修复版
4. **`yolo11_snake_bifpn.yaml`** - 原版本

### 各版本特点:
- **Simple**: 避免复杂参数，保留核心创新
- **Safe**: 只使用内置模块，确保稳定
- **Fixed**: 修复参数格式，完整功能
- **Original**: 最初版本，功能最全但可能有问题

## 🚨 紧急解决方案

如果所有自定义配置都失败，使用这个最基础的方案:

```bash
# 1. 使用标准YOLO11
python -c "from ultralytics import YOLO; model = YOLO('yolo11n.yaml'); print('Standard YOLO11 works')"

# 2. 如果标准版本工作，逐步添加自定义模块
# 先测试单个模块，再组合使用

# 3. 最后使用完整的自定义配置
```

## 📞 获取帮助

### 错误报告格式:
```
1. 运行的命令:
   python train_xray_defect.py --model xxx --data xxx

2. 完整错误信息:
   [粘贴完整的错误输出]

3. 环境信息:
   - Python版本: python --version
   - PyTorch版本: python -c "import torch; print(torch.__version__)"
   - Ultralytics版本: pip show ultralytics

4. 调试结果:
   python debug_model.py 的完整输出
```

### 常用检查命令:
```bash
# 检查ultralytics版本
pip show ultralytics

# 检查PyTorch版本
python -c "import torch; print(torch.__version__)"

# 测试标准YOLO
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# 检查YAML语法
python -c "import yaml; print(yaml.safe_load(open('models/yolo11_snake_bifpn_simple.yaml')))"
```