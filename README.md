# Local Detection Training Pipeline

**GroundedSAM / GroundingDINO → YOLOv26s** — fully local, no Colab required.

## Hardware Workflows

This pipeline supports two distinct workflows depending on your hardware:

### 1. Jetson Orin Nano (Low VRAM)
The Jetson is excellent for inference and lightweight annotation, but **cannot train YOLOv26s locally** due to its shared 8GB memory (it will run out of memory during backpropagation).
- **Goal:** Run the annotation step here, then move the dataset to your desktop.
- **Command:**
  ```bash
  python pipeline.py \
      --input_dir ./bus_dataset \
      --label bus \
      --prompt "public transit bus" \
      --annotator grounding_dino \
      --skip_training
  ```
  *(Note: `grounding_dino` uses only ~500MB VRAM compared to GroundedSAM's ~2.5GB. It runs on CPU to prevent Jetson memory crashes).*

### 2. RTX 3060 Desktop (High VRAM)
Your desktop has 12GB of dedicated VRAM, making it perfect for both annotation (using the superior GroundedSAM masks) and YOLOv26s training.
- **Goal:** Run the full end-to-end pipeline (Annotation + Training).
- **Command:**
  ```bash
  python pipeline.py \
      --input_dir ./bus_dataset \
      --label bus \
      --prompt "public transit bus" \
      --epochs 50 \
      --device 0
  ```
  *(Note: This uses the default `grounded_sam` annotator and trains using your GPU).*

## Setup

It is highly recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
*(Jetson Users: PyTorch must be the NVIDIA-compiled wheel for JetPack. Ensure your venv uses `--system-site-packages` or you manually install the JetPack PyTorch wheel).*

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--input_dir` | *(required)* | Folder of raw images to annotate |
| `--label` | `bus` | YOLO class name |
| `--prompt` | `public transit bus` | GroundedSAM text prompt |
| `--annotator` | `grounded_sam` | `grounded_sam` (DINO+SAM masks) or `grounding_dino` (bbox only) |
| `--dataset_dir` | `./dataset` | Where to save the YOLO dataset |
| `--epochs` | `50` | Training epochs |
| `--imgsz` | `640` | Training image resolution |
| `--batch` | `4` | Training batch size |
| `--workers` | `2` | DataLoader workers |
| `--device` | `0` | GPU index or `cpu` |
| `--limit` | `0` (all) | Max images to process |
| `--skip_annotation` | off | Skip auto-labeling step |
| `--skip_training` | off | Only annotate, don't train |
| `--preview` | `3` | N annotated images to save as preview (0=off) |
| `--conf` | `0.25` | Confidence threshold for validation inference |

## Output Structure

```
detection_training/
├── dataset/
│   ├── train/images/   ← annotated training images
│   ├── train/labels/   ← YOLO .txt labels
│   ├── valid/images/
│   ├── valid/labels/
│   └── data.yaml       ← dataset config
├── annotation_preview/ ← first N annotated images (visual check)
├── inference_results/  ← post-training sample predictions
└── runs/detect/train/weights/best.pt  ← final YOLOv26s weights
```
