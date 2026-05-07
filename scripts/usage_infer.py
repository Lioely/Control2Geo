#!/usr/bin/env python3
"""
Minimal single-image inference example.

Example:
    python scripts/usage_infer.py \
        --config config/model/test_bridge_control.yaml \
        --image path/to/image.jpg \
        --control path/to/control.png \
        --output_dir outputs/demo \
        --infer_steps 10 \
        --size 640 480
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.utils import create_model
from utils.infer_utils import resolve_project_path, save_infer_visualization


def parse_args():
    parser = argparse.ArgumentParser(description="Run GeoBridge inference on one image/control pair.")
    parser.add_argument("--config", default="config/model/test_bridge_control.yaml", help="Path to model config.")
    parser.add_argument("--image", required=True, help="Path to RGB input image.")
    parser.add_argument("--control", required=True, help="Path to control map. Type is inferred from model.control_key.")
    parser.add_argument("--output_dir", default="outputs/infer", help="Directory for visualization and float outputs.")
    parser.add_argument("--infer_steps", type=int, default=10, help="Number of bridge sampling steps.")
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=(640, 480),
        metavar=("HEIGHT", "WIDTH"),
        help="Inference resize in (height width) order.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="Device for inference.")
    parser.add_argument("--sample_name", default=None, help="Optional output filename prefix.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    config_path = resolve_project_path(args.config, PROJECT_ROOT)
    image_path = resolve_project_path(args.image, PROJECT_ROOT)
    control_path = resolve_project_path(args.control, PROJECT_ROOT)
    output_dir = resolve_project_path(args.output_dir, PROJECT_ROOT)

    model = create_model(str(config_path)).to(device)
    model.eval()

    with torch.no_grad():
        pred_point_map, pred_depth, pred_normal_map = model.infer(
            image_path=image_path,
            control=control_path,
            infer_steps=args.infer_steps,
            size=tuple(args.size),
        )

    sample_name = args.sample_name or image_path.stem
    saved_paths = save_infer_visualization(
        image_path=image_path,
        control_path=control_path,
        control_key=model.control_key,
        pred_point_map=pred_point_map,
        pred_depth=pred_depth,
        pred_normal_map=pred_normal_map,
        save_dir=output_dir,
        sample_name=sample_name,
    )

    print("Saved inference outputs:")
    for name, path in saved_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
