#!/usr/bin/env python3
"""Evaluate a released GeoBridge model on configured test datasets."""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.dataloader.eval_dataloader import EvalDataLoader
from model.utils import create_model
from utils.infer_utils import save_evaluation_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GeoBridge on test datasets.")
    parser.add_argument("--config", default="config/model/test_bridge_control.yaml", help="Path to model config.")
    parser.add_argument("--data_config", default="config/data/test_data.yaml", help="Path to test data config.")
    parser.add_argument("--output_dir", default="test_data", help="Directory for metrics and visualizations.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="Device for evaluation.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional maximum samples per dataset.")
    parser.add_argument("--skip_visualization", action="store_true", help="Only compute metrics.")
    return parser.parse_args()


def make_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def point_to_normal(model, point_map):
    return model.point_to_geometry(point_map)


def collect_metrics(dataset_metrics, metrics):
    for metric_type in ("points_affine_invariant", "depth_affine_invariant"):
        for metric_name in ("rel", "delta1"):
            dataset_metrics[metric_type][metric_name].append(metrics[metric_type][metric_name])


def compute_error_maps(pred_points_aligned, pred_depth_aligned, batch):
    gt_depth = batch["depth"].squeeze(1).cpu()
    gt_point_map = batch["point"].permute(0, 2, 3, 1).cpu()
    eps = 1e-6

    dist_gt = torch.norm(gt_point_map, dim=-1)
    dist_err = torch.norm(pred_points_aligned - gt_point_map, dim=-1)
    point_error = dist_err / (dist_gt + eps)
    depth_error = torch.abs(pred_depth_aligned - gt_depth) / (gt_depth + eps)
    return point_error, depth_error


def save_sample_outputs(model, output_path, sample_name, batch, pred_point_map, pred_depth, metric):
    metrics, pred_points_aligned, pred_depth_aligned = metric
    pred_normal = point_to_normal(model, pred_point_map)
    gt_normal = point_to_normal(model, batch["point"])
    point_error, depth_error = compute_error_maps(pred_points_aligned, pred_depth_aligned, batch)

    save_evaluation_outputs(
        output_dir=output_path,
        sample_name=sample_name,
        rgb_image=batch["ori_image"],
        pred_point_map=pred_point_map,
        pred_depth=pred_depth,
        pred_normal_map=pred_normal,
        gt_point_map=batch["point"],
        gt_depth=batch["depth"],
        gt_normal_map=gt_normal,
        point_error=point_error,
        depth_error=depth_error,
        mask=batch["depth_mask"],
    )
    return metrics


def evaluate_dataset(model, dataset_name, dataloader, output_root, max_samples=None, skip_visualization=False):
    print(f"Evaluating dataset {dataset_name}")
    dataset_metrics = defaultdict(lambda: defaultdict(list))
    dataset_output = output_root / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)

    for sample_idx, batch in enumerate(tqdm(dataloader)):
        if max_samples is not None and sample_idx >= max_samples:
            break

        with torch.no_grad():
            pred_point_map, pred_depth, metric = model.evaluate_infer(batch)

        metrics = metric[0]
        collect_metrics(dataset_metrics, metrics)

        if not skip_visualization:
            save_sample_outputs(
                model=model,
                output_path=dataset_output,
                sample_name=str(sample_idx),
                batch=batch,
                pred_point_map=pred_point_map,
                pred_depth=pred_depth,
                metric=metric,
            )

    return dataset_metrics


def summarize_dataset(dataset_name, dataset_metrics, output_root):
    print(f"\n{'=' * 50}")
    print(f"Results for {dataset_name}")
    print(f"{'=' * 50}")

    dataframe = pd.DataFrame()
    summary = {}
    for metric_type in ("points_affine_invariant", "depth_affine_invariant"):
        print(f"\n{metric_type}:")
        for metric_name in ("rel", "delta1"):
            values = [
                float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
                for value in dataset_metrics[metric_type][metric_name]
            ]
            mean_value = float(np.mean(values)) if values else float("nan")
            dataframe[f"{metric_type}_{metric_name}"] = values
            summary[f"{metric_type}_{metric_name}"] = mean_value
            print(f"  {metric_name}: {mean_value:.4f}")

    dataframe.to_csv(output_root / f"{dataset_name}.csv", index=False)
    print(f"{'=' * 50}\n")
    return summary


def main():
    args = parse_args()
    device = make_device(args.device)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    model = create_model(args.config).to(device)
    model.eval()

    eval_dataset = EvalDataLoader(args.data_config)
    all_metrics = defaultdict(list)

    for dataset_name, dataloader in zip(eval_dataset.dataset_list, eval_dataset.dataloader_ls):
        dataset_metrics = evaluate_dataset(
            model=model,
            dataset_name=dataset_name,
            dataloader=dataloader,
            output_root=output_root,
            max_samples=args.max_samples,
            skip_visualization=args.skip_visualization,
        )
        summary = summarize_dataset(dataset_name, dataset_metrics, output_root)
        for metric_name, value in summary.items():
            all_metrics[metric_name].append(value)

    pd.DataFrame(all_metrics).to_csv(output_root / "all_metrics.csv", index=False, encoding="utf-8")
    print(f"Metrics saved to {output_root / 'all_metrics.csv'}")


if __name__ == "__main__":
    main()
