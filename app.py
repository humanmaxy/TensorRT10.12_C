import os
import time
import glob
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

# Import custom modules for enhanced attention mechanisms
import custom_modules  # This automatically registers our custom modules

# Paths
WORKDIR = Path(__file__).resolve().parent
DEFAULT_MODEL_YAML = WORKDIR / 'models' / 'yolo11_surface_defect_p2.yaml'
DEFAULT_DATA_YAML = WORKDIR / 'data' / 'surface_defect.yaml'
RUNS_DIR = WORKDIR / 'runs' / 'train'

# Ensure directories exist
(WORKDIR / 'models').mkdir(parents=True, exist_ok=True)
(WORKDIR / 'data').mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title='YOLO11 Surface Defect Trainer', layout='wide')
st.title('YOLOv11 可视化配置训练器 - 表面缺陷检测')

with st.sidebar:
	st.header('模型与数据配置')
	# Model variants and blocks
	scale = st.selectbox('Model Scale (n/s/m/l/x)', ['n','s','m','l','x'], index=0)
	use_p2 = st.checkbox('启用P2输出（更适合微小缺陷）', value=True)
	use_c2psa = st.checkbox('启用C2PSA注意力', value=True)
	use_coordatt = st.checkbox('启用Coordinate Attention (提升MAP)', value=True)
	backbone_block = st.selectbox('Backbone模块类型', ['C3k2','C2f'], index=0)
	head_block = st.selectbox('Head模块类型', ['C3k2','C2f'], index=0)
	nc = st.number_input('类别数 (nc)', min_value=1, max_value=200, value=3, step=1)

	# Data paths
	dataset_root = st.text_input('数据集根路径 path', str(WORKDIR / 'datasets' / 'surface_defect'))
	train_rel = st.text_input('训练集相对路径', 'images/train')
	val_rel = st.text_input('验证集相对路径', 'images/val')
	test_rel = st.text_input('测试集相对路径 (可选)', 'images/test')
	labels_train_rel = st.text_input('训练标签相对路径 (可选)', 'labels/train')
	labels_val_rel = st.text_input('验证标签相对路径 (可选)', 'labels/val')
	class_names_str = st.text_input('类别名（逗号分隔）', 'scratch,dent,crack')

	st.header('训练配置')
	imgsz = st.slider('图像尺寸', 448, 1280, 640, 32)
	epochs = st.slider('训练轮数', 10, 500, 200, 10)
	batch = st.slider('批大小', 1, 32, 4, 1)
	workers = st.slider('DataLoader workers', 0, 8, 0, 1)
	optimizer = st.selectbox('优化器', ['SGD','Adam','AdamW','auto'], index=0)
	cos_lr = st.checkbox('Cosine LR', value=True)
	device = st.text_input('设备', '0')
	default_name = f'yolo11-surface-p2-coordatt-{int(time.time())}' if use_coordatt else f'yolo11-surface-p2-{int(time.time())}'
	exp_name = st.text_input('实验名', default_name)

	st.header('预训练权重（离线）')
	st.caption('离线模式：不下载任何权重。可选填写本地 .pt 路径，否则从零训练。')
	custom_weights = st.text_input('本地预训练权重路径(.pt，可留空)', '')

	start_train = st.button('开始训练', type='primary')

# Load base model YAML text (custom only to avoid heavy imports); otherwise use YAML fallback later
if use_coordatt:
	# Use enhanced model with Coordinate Attention
	ENHANCED_MODEL_YAML = WORKDIR / 'models' / 'yolo11_surface_defect_p2_coordatt_final.yaml'
	model_yaml_text = ENHANCED_MODEL_YAML.read_text() if ENHANCED_MODEL_YAML.exists() else ''
	if not model_yaml_text:
		st.error("Enhanced CoordAtt model config not found! Using default model.")
		model_yaml_text = DEFAULT_MODEL_YAML.read_text() if DEFAULT_MODEL_YAML.exists() else ''
else:
	model_yaml_text = DEFAULT_MODEL_YAML.read_text() if DEFAULT_MODEL_YAML.exists() else ''

model_source = None
if model_yaml_text:
	# replace nc and C2PSA toggle first
	model_yaml_text = model_yaml_text.replace('nc: 80', f'nc: {nc}')
	if not use_c2psa:
		model_yaml_text = model_yaml_text.replace('C2PSA', 'C3k2')
	# swap backbone/head block names
	lines = model_yaml_text.splitlines()
	try:
		idx_backbone = next(i for i, l in enumerate(lines) if l.strip().startswith('backbone:'))
		idx_head = next(i for i, l in enumerate(lines) if l.strip().startswith('head:'))
		b_lines = lines[idx_backbone + 1:idx_head]
		h_lines = lines[idx_head + 1:]
		if backbone_block != 'C3k2':
			if use_coordatt:
				# For CoordAtt model, replace enhanced versions
				b_lines = [ln.replace('C3k2_CoordAtt', f'{backbone_block}_CoordAtt') for ln in b_lines]
				b_lines = [ln.replace('C3k2', backbone_block) for ln in b_lines]
			else:
				b_lines = [ln.replace('C3k2', backbone_block) for ln in b_lines]
		if head_block != 'C3k2':
			if use_coordatt:
				# For CoordAtt model, replace enhanced versions
				h_lines = [ln.replace('C3k2_CoordAtt', f'{head_block}_CoordAtt') for ln in h_lines]
				h_lines = [ln.replace('C3k2', head_block) for ln in h_lines]
			else:
				h_lines = [ln.replace('C3k2', head_block) for ln in h_lines]
		lines = lines[:idx_backbone + 1] + b_lines + [lines[idx_head]] + h_lines
		model_yaml_text = '\n'.join(lines)
	except StopIteration:
		pass
	# Detect heads selection for P2 toggle (only if our custom four-head exists)
	if not use_p2:
		model_yaml_text = model_yaml_text.replace('[[19, 22, 25, 28], 1, Detect, [nc]]', '[[22, 25, 28], 1, Detect, [nc]]')
	# scale selection: map to 'n'
	scales_map = {'n':'0.50, 0.25, 1024','s':'0.50, 0.50, 1024','m':'0.50, 1.00, 512','l':'1.00, 1.00, 512','x':'1.00, 1.50, 512'}
	model_yaml_text = model_yaml_text.replace('n: [0.50, 0.25, 1024]', f"n: [{scales_map[scale]}]")
	# write temp model yaml
	temp_model_yaml = WORKDIR / 'models' / f'_tmp_{scale}_model.yaml'
	temp_model_yaml.parent.mkdir(parents=True, exist_ok=True)
	temp_model_yaml.write_text(model_yaml_text)
	model_source = str(temp_model_yaml)
else:
	# final fallback to local YAML name (no download)
	model_source = 'yolo11.yaml'

# Create data YAML dynamically
names = [x.strip() for x in class_names_str.split(',') if x.strip()]
quoted_names = ', '.join([f"'{n}'" for n in names])
lines = [
	f"path: {dataset_root}",
	f"train: {train_rel}",
	f"val: {val_rel}",
]
if test_rel:
	lines.append(f"test: {test_rel}")
lines += [
	f"nc: {nc}",
	f"names: [{quoted_names}]",
]
data_yaml_text = "\n".join(lines) + "\n"
temp_data_yaml = WORKDIR / 'data' / f'_tmp_data_{int(time.time())}.yaml'
temp_data_yaml.parent.mkdir(parents=True, exist_ok=True)
temp_data_yaml.write_text(data_yaml_text)

# Preview panel
col1, col2 = st.columns(2)
with col1:
	st.subheader('模型配置预览')
	st.code(model_yaml_text if model_yaml_text else 'Using YAML: ' + str(model_source), language='yaml')
with col2:
	st.subheader('数据配置预览')
	st.code(data_yaml_text, language='yaml')

# Start training
log_placeholder = st.empty()
progress = st.empty()
metrics_placeholder = st.empty()
image_preview_placeholder = st.empty()

if start_train:
	# Preflight dataset path checks to avoid runtime errors
	images_train_dir = Path(dataset_root) / train_rel
	images_val_dir = Path(dataset_root) / val_rel
	labels_train_dir = Path(dataset_root) / labels_train_rel if labels_train_rel else Path(dataset_root) / train_rel.replace('images', 'labels')
	labels_val_dir = Path(dataset_root) / labels_val_rel if labels_val_rel else Path(dataset_root) / val_rel.replace('images', 'labels')
	missing = []
	for p in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
		if not p.exists():
			missing.append(str(p))
	if missing:
		st.error('以下路径不存在，请在侧边栏修正数据集根目录或相对路径后再开始训练:\n' + '\n'.join(missing))
		st.stop()
	# Lazy import to avoid DLL load until training
	from ultralytics import YOLO
	st.toast('开始训练...', icon='✅')
	model = YOLO(model_source)
	# Load custom pretrained weights if provided (offline local only)
	if custom_weights.strip():
		model.load(custom_weights.strip())
	# Always disable built-in pretrained to avoid online download in offline mode
	kwargs = dict(
		data=str(temp_data_yaml),
		imgsz=imgsz,
		epochs=epochs,
		batch=batch,
		workers=workers,
		optimizer=optimizer,
		cos_lr=cos_lr,
		device=device,
		project=str(RUNS_DIR),
		name=exp_name,
		pretrained=False,
	)
	results = model.train(**kwargs)
	# After training, parse results.csv
	run_dir = RUNS_DIR / exp_name
	csv_path = run_dir / 'results.csv'
	if csv_path.exists():
		df = pd.read_csv(csv_path)
		metrics_placeholder.subheader('训练曲线')
		for key in ['metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)','train/box_loss','train/cls_loss','train/dfl_loss']:
			if key in df.columns:
				fig = px.line(df, x=df.index, y=key, title=key)
				metrics_placeholder.plotly_chart(fig, use_container_width=True)
		# Show preview images from val
		pred_dir = run_dir / 'val' / 'predictions'
		if pred_dir.exists():
			imgs = sorted(glob.glob(str(pred_dir / '*.jpg')))[:8]
			image_preview_placeholder.subheader('验证集预测预览')
			if imgs:
				cols = st.columns(4)
				for i, p in enumerate(imgs):
					cols[i % 4].image(p, use_container_width=True)
			else:
				image_preview_placeholder.info('未找到预测图片')
	else:
		st.error('训练日志未找到，可能训练失败或目录结构变化')