---

# 🛰️ Remote Sensing + Object Detection with YOLOv8
---
This repository contains tools and workflows for a remote sensing research project combining:
- Satellite image processing
- Object detection using YOLOv8
- Annotation format conversion, KITTI -> YOLO
- Training and evaluation of deep learning models on supercomputing infrastructure

---

## 📌 Project Overview

We use high-resolution satellite imagery (Sentinel, Landsat, etc.) to detect and classify features like ancient raised fields, lakes, roads, and human settlements. This involves:

- Data annotation + format conversion
- Custom YOLOv8 model training
- Feature detection and typology
- 3D visualization and slicing (SAHI)

---

## 🧭 Core Workflows

### 1. 🗂 KITTI to YOLOv8 Label Conversion

Use the `convert_kitti_to_yolo()` function to:
- Normalize KITTI coordinates
- Compute YOLO bounding box format
- Validate, clean, and save results

Output includes:
- Converted labels
- Statistical summaries (mean/variance of dimensions)
- Warnings for malformed inputs
- Optional visualization of bounding boxes

**Script entry point:**  
```bash
python convert_kitti.py
```

---

### 2. 🧠 YOLOv8 Training & Inference Pipeline

Training pipeline:
- Uses `ultralytics` YOLOv8 implementation
- Trains on custom satellite imagery
- Integrates TensorBoard for visualization
- Supports sliced image prediction with [SAHI](https://github.com/obss/sahi)

```python
# Training
model = YOLO('yolov8n.pt')
model.train(data=DATA_YAML, epochs=120, imgsz=640)

# Predict (SAHI slicing)
predict_with_sahi(best_model_path, image_folder)
```

> Output directories follow YOLO's `runs/detect/` structure.

---

### 3. 🧪 Validation & Visualization

The repo includes:
- Random sample visualization (`plot()`)
- Grid-based UIs using `ImageGrid`
- Dataset integrity checker for:
  - Label/image mismatch
  - Empty label files
  - Missing annotations
- Grid visualizer to preview final detections

---

## 🔬 Tools & Libraries Used

- [`ultralytics`](https://github.com/ultralytics/ultralytics) (YOLOv8)
- [`SAHI`](https://github.com/obss/sahi) for slicing large satellite tiles
- `OpenCV`, `Matplotlib`, `TQDM`, `PIL`
- `KITTI` → `YOLO` custom converter
- `TensorBoard` for training diagnostics

---

## 📁 Project Structure

```bash
/convert_kitti.py           # Label format conversion + validation
/train_yolo_pipeline.py     # Main training + prediction script
/yolo_helpers.py            # Plotting, validation, and utilities

/dataRR/
  ├── imagesRR/             # Raw training images
  ├── labelsRR3/            # KITTI format annotations
  ├── validRR/              # YOLO-processed labels
  ├── data.yaml             # YOLOv8 dataset config

/runs/
  └── detect/               # YOLOv8 output directory
```

---

## ✅ Sample Output (Conversion Stats)

```
Total files processed: 5408
Total boxes converted: 93722
Invalid values found: 0

width_stats:  Mean: 0.1091  Max: 1.0000
height_stats: Mean: 0.1123  Max: 1.0000
```

---

## 🚀 Getting Started

### Requirements
```bash
pip install ultralytics sahi opencv-python matplotlib pandas tqdm
```

### Run the full pipeline:
1. Convert annotations:
   ```bash
   python convert_kitti.py
   ```
2. Train model:
   ```bash
   python train_yolo_pipeline.py
   ```
3. Predict with slicing:
   ```bash
   python train_yolo_pipeline.py --predict
   ```

---

## 🔍 Planned Enhancements

- [ ] Add support for Pascal VOC and COCO conversion
- [ ] Extend visualization to show false positives/negatives
- [ ] Create GUI-based annotation checker (Gradio)
- [ ] Publish a Jupyter-based dashboard for training metrics

---

## 📄 License

MIT License – see `LICENSE` file for details.

---

## ✉️ Contact

Maintainer: [Your Name]  
Institution: [Your Lab / Department / Institution]  
Email: [your.email@domain.com]

---

Let me know if you'd like me to also generate a `requirements.txt`, `.gitignore`, or `LICENSE` file for this repo as I did earlier — I can create those tailored to this pipeline too.
