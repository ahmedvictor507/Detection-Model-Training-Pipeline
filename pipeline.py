#!/usr/bin/env python3
"""
Local Object Detection Training Pipeline
=========================================
Auto-annotates images with GroundedSAM, then trains YOLOv26s.

Usage:
    python pipeline.py --input_dir ./my_images --label "bus" --prompt "public transit bus" --epochs 50

Steps:
    1. Reads images from --input_dir
    2. Auto-annotates with GroundedSAM (Autodistill)
    3. Trains YOLOv26s on the generated dataset
    4. Runs a quick validation inference on 3 sample images
"""

import argparse
import os
import shutil
import glob
import cv2
import sys

# ──────────────────────────────────────────────────────────────────────────────
# CLI ARGUMENTS
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Local GroundedSAM → YOLOv26s training pipeline"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to folder containing raw images (jpg/png)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="bus",
        help="Target YOLO class name (e.g. 'bus', 'car', 'person')",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="public transit bus",
        help="English text prompt for GroundedSAM (e.g. 'public transit bus')",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./dataset",
        help="Where to save the annotated YOLO dataset (default: ./dataset)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of YOLOv26s training epochs (default: 50)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image resolution (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Training device: '0' for GPU, 'cpu' for CPU (default: '0')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of images to process (0 = all images, default: 0)",
    )
    parser.add_argument(
        "--skip_annotation",
        action="store_true",
        help="Skip annotation step if dataset already exists",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training step (only annotate)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        help="Number of annotated images to preview after labeling (0 = skip, default: 3)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for validation inference (default: 0.25)",
    )
    parser.add_argument(
        "--annotator",
        type=str,
        default="grounded_sam",
        choices=["grounded_sam", "grounding_dino"],
        help=(
            "Auto-labeling backend (default: grounded_sam).\n"
            "  grounded_sam  — full DINO+SAM, ~2.5GB VRAM, best masks (RTX 3060 / desktop GPU)\n"
            "  grounding_dino — bbox-only,   ~500MB VRAM, lighter (Jetson / low-memory)"
        ),
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Training batch size (default: 4, keep low for Jetson)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Dataloader workers (default: 2, keep low for Jetson)",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 – COLLECT IMAGES
# ──────────────────────────────────────────────────────────────────────────────

def collect_images(input_dir: str, limit: int = 0) -> list[str]:
    """Return a list of image paths from input_dir, optionally capped."""
    input_dir = os.path.abspath(input_dir)
    if not os.path.isdir(input_dir):
        print(f"❌  Input directory not found: {input_dir}")
        sys.exit(1)

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

    # Deduplicate (case-insensitive on ext) and sort for reproducibility
    images = sorted(set(images))

    if not images:
        print(f"❌  No images found inside: {input_dir}")
        sys.exit(1)

    if limit and limit > 0:
        images = images[:limit]

    print(f"🖼️   Found {len(images)} image(s) in '{input_dir}'")
    return images


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 – AUTO-ANNOTATE WITH GROUNDEDSAM
# ──────────────────────────────────────────────────────────────────────────────

def run_annotation(
    images: list[str],
    label: str,
    prompt: str,
    dataset_dir: str,
    annotator: str = "grounded_sam",
):
    """Auto-label images and write a YOLO-format dataset to dataset_dir.

    annotator options:
        'grounded_sam'   — GroundingDINO + SAM masks, ~2.5GB VRAM (RTX 3060 / desktop)
        'grounding_dino' — bounding boxes only,        ~500MB VRAM (Jetson / low-memory)
    """
    from autodistill.detection import CaptionOntology

    if annotator == "grounded_sam":
        try:
            from autodistill_grounded_sam import GroundedSAM as BaseModel
            backend_name = "GroundedSAM (DINO + SAM masks)"
        except Exception as e:
            import traceback
            print(f"❌  Failed to import autodistill-grounded-sam:")
            print(f"    {type(e).__name__}: {e}")
            print(f"    Install: pip install autodistill-grounded-sam")
            traceback.print_exc()
            sys.exit(1)
    else:  # grounding_dino
        try:
            from autodistill_grounding_dino import GroundingDINO as BaseModel
            backend_name = "GroundingDINO (bbox-only, low-VRAM)"
        except Exception as e:
            import traceback
            print(f"❌  Failed to import autodistill-grounding-dino:")
            print(f"    {type(e).__name__}: {e}")
            print(f"    Install: pip install autodistill-grounding-dino")
            traceback.print_exc()
            sys.exit(1)

    # Build a staging directory so Autodistill gets a clean image folder
    staging_dir = "./staging_images"
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)

    print(f"\n📋  Staging {len(images)} image(s) for annotation (resizing to ≤640px for Jetson)...")
    for i, src in enumerate(images):
        ext = os.path.splitext(src)[1].lower() or ".jpg"
        dst = os.path.join(staging_dir, f"img_{i:05d}{ext}")
        # Resize to max 640px on the longest side to keep VRAM usage low on Jetson
        img = cv2.imread(src)
        if img is not None:
            h, w = img.shape[:2]
            max_side = 640
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dst, img)
        else:
            shutil.copy2(src, dst)  # fallback: copy as-is

    ontology = CaptionOntology({prompt: label})
    base_model = BaseModel(ontology=ontology)

    print(f"🤖  Running {backend_name}  [ prompt: '{prompt}' → class: '{label}' ]")
    print(f"    Output dataset : {os.path.abspath(dataset_dir)}")

    base_model.label(
        input_folder=staging_dir,
        output_folder=dataset_dir,
    )

    # Clean up staging area
    shutil.rmtree(staging_dir)
    print(f"\n✅  Annotation complete! Dataset saved to: {os.path.abspath(dataset_dir)}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 – PREVIEW ANNOTATIONS (optional)
# ──────────────────────────────────────────────────────────────────────────────

def preview_annotations(dataset_dir: str, n: int = 3):
    """Display the first n annotated images using supervision + OpenCV."""
    try:
        import supervision as sv
    except ImportError:
        print("⚠️   supervision not installed, skipping preview.")
        return

    images_dir = os.path.join(dataset_dir, "train", "images")
    labels_dir = os.path.join(dataset_dir, "train", "labels")
    data_yaml  = os.path.join(dataset_dir, "data.yaml")

    if not os.path.isdir(images_dir):
        print(f"⚠️   Preview skipped – train/images not found in {dataset_dir}")
        return

    try:
        dataset = sv.DetectionDataset.from_yolo(
            images_directory_path=images_dir,
            annotations_directory_path=labels_dir,
            data_yaml_path=data_yaml,
        )
    except Exception as e:
        print(f"⚠️   Could not load dataset for preview: {e}")
        return

    print(f"\n👁️   Previewing {min(n, len(dataset))} annotation(s)...")
    print(f"    Classes: {dataset.classes}")

    box_ann   = sv.BoxAnnotator()
    label_ann = sv.LabelAnnotator()
    preview_out = "./annotation_preview"
    os.makedirs(preview_out, exist_ok=True)

    count = 0
    for image_path, image, detections in dataset:
        if count >= n:
            break

        labels = [dataset.classes[cid] for cid in detections.class_id]
        annotated = box_ann.annotate(scene=image.copy(), detections=detections)
        annotated = label_ann.annotate(scene=annotated, detections=detections, labels=labels)

        out_path = os.path.join(preview_out, f"preview_{count:03d}.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"    [{count + 1}] {image_path}  →  saved to {out_path}")
        count += 1

    print(f"✅  Previews saved to: {os.path.abspath(preview_out)}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 – TRAIN YOLOv26s
# ──────────────────────────────────────────────────────────────────────────────

def run_training(
    dataset_dir: str,
    epochs: int,
    imgsz: int,
    device: str,
    batch: int = 4,
    workers: int = 2,
) -> str:
    """Train YOLOv26s and return the path to best.pt."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌  ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    data_yaml = os.path.join(dataset_dir, "data.yaml")
    if not os.path.isfile(data_yaml):
        print(f"❌  data.yaml not found at {data_yaml}")
        print("    Run annotation step first (remove --skip_annotation).")
        sys.exit(1)

    print(f"\n🚀  Starting YOLOv26s training")
    print(f"    Model   : yolo26s.pt")
    print(f"    Data    : {os.path.abspath(data_yaml)}")
    print(f"    Epochs  : {epochs}")
    print(f"    Img size: {imgsz}")
    print(f"    Batch   : {batch}")
    print(f"    Workers : {workers}")
    print(f"    Device  : {device}")

    # device argument: YOLO expects an int for CUDA or the string 'cpu'
    device_arg = int(device) if device.isdigit() else device

    model = YOLO("yolo26s.pt")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        device=device_arg,
        batch=batch,
        workers=workers,
    )

    # Locate best weights produced by the run
    best_pt = os.path.join(str(results.save_dir), "weights", "best.pt")
    if not os.path.isfile(best_pt):
        # Fallback: search for it
        candidates = glob.glob("runs/detect/**/weights/best.pt", recursive=True)
        best_pt = candidates[-1] if candidates else "best.pt"

    print(f"\n🏆  Training complete!")
    print(f"    Best weights: {os.path.abspath(best_pt)}")
    return best_pt


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 – VALIDATION INFERENCE
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(
    best_pt: str,
    dataset_dir: str,
    conf: float = 0.25,
    n: int = 3,
):
    """Run inference on n validation images and save the results."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return

    if not os.path.isfile(best_pt):
        print(f"⚠️   Weights file not found: {best_pt}. Skipping inference.")
        return

    val_dir = os.path.join(dataset_dir, "valid", "images")
    if not os.path.isdir(val_dir):
        val_dir = os.path.join(dataset_dir, "train", "images")

    images = glob.glob(os.path.join(val_dir, "*.jpg")) + \
             glob.glob(os.path.join(val_dir, "*.png"))
    images = sorted(images)[:n]

    if not images:
        print("⚠️   No images found for validation inference.")
        return

    print(f"\n🔍  Running inference on {len(images)} sample(s)  [conf ≥ {conf}]")
    model = YOLO(best_pt)
    out_dir = "./inference_results"
    os.makedirs(out_dir, exist_ok=True)

    for i, img_path in enumerate(images):
        results = model(img_path, conf=conf)
        annotated = results[0].plot()
        out_path = os.path.join(out_dir, f"result_{i:03d}.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"    [{i + 1}] {img_path}  →  {out_path}")

    print(f"✅  Inference results saved to: {os.path.abspath(out_dir)}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  Local Detection Pipeline  |  GroundedSAM → YOLOv26s")
    print("=" * 60)

    # ── Step 1: Collect images ────────────────────────────────────────────────
    images = collect_images(args.input_dir, limit=args.limit)

    # ── Step 2: Annotate ──────────────────────────────────────────────────────
    if not args.skip_annotation:
        run_annotation(
            images=images,
            label=args.label,
            prompt=args.prompt,
            dataset_dir=args.dataset_dir,
            annotator=args.annotator,
        )
    else:
        print("\n⏩  Annotation skipped (--skip_annotation flag set).")

    # ── Step 3: Preview ───────────────────────────────────────────────────────
    if args.preview > 0:
        preview_annotations(args.dataset_dir, n=args.preview)

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    best_pt = None
    if not args.skip_training:
        best_pt = run_training(
            dataset_dir=args.dataset_dir,
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=args.device,
            batch=args.batch,
            workers=args.workers,
        )
    else:
        print("\n⏩  Training skipped (--skip_training flag set).")

    # ── Step 5: Validation inference ─────────────────────────────────────────
    if best_pt:
        run_inference(
            best_pt=best_pt,
            dataset_dir=args.dataset_dir,
            conf=args.conf,
            n=args.preview if args.preview > 0 else 3,
        )

    print("\n" + "=" * 60)
    print("  Pipeline finished successfully! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    main()
