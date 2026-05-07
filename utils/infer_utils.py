import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# OpenCV checks this flag when its EXR codec is initialized.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
from PIL import Image

from utils.vis import colorize_depth_affine


PathLike = Union[str, os.PathLike]


def resolve_project_path(path_value: PathLike, project_root: Optional[PathLike] = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    root = Path.cwd() if project_root is None else Path(project_root)
    return (root / path).resolve()


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def squeeze_batch(array: Any) -> np.ndarray:
    array = tensor_to_numpy(array)
    if array.ndim > 0 and array.shape[0] == 1:
        return np.squeeze(array, axis=0)
    return array


def to_uint8_image(array: Any) -> np.ndarray:
    array = np.nan_to_num(tensor_to_numpy(array))
    min_value = array.min()
    max_value = array.max()
    if max_value - min_value < 1e-8:
        return np.zeros_like(array, dtype=np.uint8)
    return ((array - min_value) / (max_value - min_value) * 255.0).astype(np.uint8)


def colorize_depth_safe(depth: Any, mask: Optional[Any] = None) -> np.ndarray:
    depth = np.nan_to_num(tensor_to_numpy(depth).astype(np.float32))
    mask_array = None if mask is None else tensor_to_numpy(mask).astype(bool)
    valid = np.isfinite(depth) if mask_array is None else mask_array & np.isfinite(depth)

    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    valid_values = depth[valid]
    if valid_values.max() - valid_values.min() < 1e-8:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    return colorize_depth_affine(depth, mask=valid)


def point_map_to_uint8(point_map: Any) -> np.ndarray:
    point_map = squeeze_batch(point_map)
    if point_map.ndim == 3 and point_map.shape[0] == 3:
        point_map = np.transpose(point_map, (1, 2, 0))
    return to_uint8_image(point_map)


def normal_to_uint8(normal: Any, flip_yz: bool = False) -> np.ndarray:
    normal = squeeze_batch(normal)
    if normal.ndim == 3 and normal.shape[0] == 3:
        normal = np.transpose(normal, (1, 2, 0))
    normal = np.nan_to_num(normal.astype(np.float32))
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6)
    if flip_yz:
        normal = normal * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    return ((np.clip(normal, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)


def save_float_output(
    exr_path: PathLike,
    array: Any,
    convert_rgb_to_bgr: bool = False,
    exr_type: Optional[int] = None,
) -> Path:
    array = tensor_to_numpy(array).astype(np.float32)
    exr_path = Path(exr_path)
    exr_path.parent.mkdir(parents=True, exist_ok=True)

    if exr_type is None:
        exr_type = cv2.IMWRITE_EXR_TYPE_FLOAT

    data_to_write = cv2.cvtColor(array, cv2.COLOR_RGB2BGR) if convert_rgb_to_bgr else array
    try:
        ok = cv2.imwrite(str(exr_path), data_to_write, [cv2.IMWRITE_EXR_TYPE, exr_type])
    except cv2.error as exc:
        ok = False
        print(f"OpenCV EXR save failed for {exr_path.name}: {exc}")

    if ok:
        return exr_path

    fallback_path = exr_path.with_suffix(".npy")
    np.save(fallback_path, array)
    print(f"Saved fallback numpy array: {fallback_path}")
    return fallback_path


def read_control_array(control_path: PathLike) -> np.ndarray:
    control_path = Path(control_path)
    suffix = control_path.suffix.lower()

    if suffix == ".npy":
        return np.load(control_path).squeeze()

    if suffix == ".exr":
        control = cv2.imread(str(control_path), cv2.IMREAD_UNCHANGED)
        if control is None:
            raise FileNotFoundError(f"Cannot read control map: {control_path}")
        if control.ndim == 3:
            control = cv2.cvtColor(control, cv2.COLOR_BGR2RGB)
        return control

    return np.asarray(Image.open(control_path))


def resize_control_array(control: np.ndarray, width: int, height: int, is_edge: bool = False) -> np.ndarray:
    interpolation = cv2.INTER_NEAREST if is_edge else cv2.INTER_LINEAR
    return cv2.resize(control, (width, height), interpolation=interpolation)


def visualize_normal_control(control: Any) -> np.ndarray:
    control = tensor_to_numpy(control)
    if control.ndim == 2:
        control = np.repeat(control[..., None], 3, axis=-1)
    if control.ndim == 3 and control.shape[-1] > 3:
        control = control[..., :3]

    control = np.nan_to_num(control.astype(np.float32))
    min_value = control.min()
    max_value = control.max()
    if min_value < 0.0:
        control = (np.clip(control, -1.0, 1.0) + 1.0) * 127.5
    elif max_value <= 1.0:
        control = control * 255.0
    return np.clip(control, 0.0, 255.0).astype(np.uint8)


def visualize_depth_control(control: Any) -> np.ndarray:
    control = tensor_to_numpy(control)
    if control.ndim == 3:
        if control.shape[-1] >= 3:
            control = cv2.cvtColor(control[..., :3].astype(np.float32), cv2.COLOR_RGB2GRAY)
        else:
            control = control[..., 0]
    return colorize_depth_safe(control.astype(np.float32))


def visualize_edge_control(control: Any) -> np.ndarray:
    control = tensor_to_numpy(control)
    if control.ndim == 3:
        control = cv2.cvtColor(control[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    control = to_uint8_image(control)
    return np.repeat(control[..., None], 3, axis=-1)


def visualize_control_map(
    control_path: PathLike,
    control_key: str,
    width: int,
    height: int,
) -> Tuple[np.ndarray, str]:
    control_key = str(control_key).lower()
    control = read_control_array(control_path)
    control = resize_control_array(control, width, height, is_edge=control_key in ("edge", "canny"))

    if control_key == "normal":
        return visualize_normal_control(control), "normal"
    if control_key in ("edge", "canny"):
        return visualize_edge_control(control), "edge"
    if control_key == "depth":
        return visualize_depth_control(control), "depth"

    if control.ndim == 3:
        return np.asarray(control[..., :3], dtype=np.uint8), control_key
    return np.repeat(to_uint8_image(control)[..., None], 3, axis=-1), control_key


def read_rgb_image(image_path: PathLike) -> np.ndarray:
    return np.asarray(Image.open(image_path).convert("RGB"))


def save_infer_visualization(
    image_path: PathLike,
    control_path: PathLike,
    control_key: str,
    pred_point_map: Any,
    pred_depth: Any,
    pred_normal_map: Any,
    save_dir: PathLike,
    sample_name: str = "sample",
) -> Dict[str, Path]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rgb = read_rgb_image(image_path)
    height, width = rgb.shape[:2]

    control_vis, control_type = visualize_control_map(control_path, control_key, width, height)
    point_hw3 = squeeze_batch(pred_point_map)
    depth_hw = squeeze_batch(pred_depth)
    normal_vis = normal_to_uint8(pred_normal_map)

    if point_hw3.ndim == 3 and point_hw3.shape[0] == 3:
        point_hw3 = np.transpose(point_hw3, (1, 2, 0))

    point_vis = point_map_to_uint8(point_hw3)
    depth_vis = colorize_depth_safe(depth_hw)

    vis_path = save_dir / f"{sample_name}_vis.png"
    control_path_out = save_dir / f"{sample_name}_control_{control_type}.png"
    Image.fromarray(control_vis).save(control_path_out)
    Image.fromarray(np.concatenate([rgb, control_vis, point_vis, depth_vis, normal_vis], axis=1)).save(vis_path)

    points_path = save_float_output(save_dir / f"{sample_name}_points.exr", point_hw3, convert_rgb_to_bgr=True)
    depth_path = save_float_output(save_dir / f"{sample_name}_depth.exr", depth_hw)

    return {
        "vis": vis_path,
        "control": control_path_out,
        "points": points_path,
        "depth": depth_path,
    }


def save_evaluation_outputs(
    output_dir: PathLike,
    sample_name: str,
    rgb_image: Any,
    pred_point_map: Any,
    pred_depth: Any,
    pred_normal_map: Any,
    gt_point_map: Optional[Any] = None,
    gt_depth: Optional[Any] = None,
    gt_normal_map: Optional[Any] = None,
    point_error: Optional[Any] = None,
    depth_error: Optional[Any] = None,
    mask: Optional[Any] = None,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    for subdir in ("vis", "depth", "points", "normal", "img", "mask"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    rgb = squeeze_batch(rgb_image)
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip((rgb + 1.0) / 2.0 if rgb.dtype.kind == "f" and rgb.min() < 0 else rgb, 0, 1 if rgb.dtype.kind == "f" else 255)
    rgb_uint8 = (rgb * 255.0).astype(np.uint8) if rgb.dtype.kind == "f" else rgb.astype(np.uint8)

    pred_points = squeeze_batch(pred_point_map)
    if pred_points.ndim == 3 and pred_points.shape[0] == 3:
        pred_points = np.transpose(pred_points, (1, 2, 0))
    pred_depth_hw = squeeze_batch(pred_depth)
    pred_normal_vis = normal_to_uint8(pred_normal_map, flip_yz=True)

    rows = []
    if gt_point_map is not None and gt_depth is not None and gt_normal_map is not None:
        gt_points_vis = point_map_to_uint8(gt_point_map)
        gt_depth_vis = colorize_depth_safe(squeeze_batch(gt_depth), mask=squeeze_batch(mask) if mask is not None else None)
        gt_normal_vis = normal_to_uint8(gt_normal_map, flip_yz=True)
        rows.append(np.concatenate([rgb_uint8, gt_points_vis, gt_depth_vis, gt_normal_vis], axis=1))

    blank = np.full_like(rgb_uint8, 255, dtype=np.uint8)
    pred_row = np.concatenate(
        [
            blank,
            point_map_to_uint8(pred_points),
            colorize_depth_safe(pred_depth_hw),
            pred_normal_vis,
        ],
        axis=1,
    )
    rows.append(pred_row)

    if point_error is not None and depth_error is not None:
        rows.append(
            np.concatenate(
                [
                    blank,
                    colorize_depth_safe(squeeze_batch(point_error), mask=squeeze_batch(mask) if mask is not None else None),
                    colorize_depth_safe(squeeze_batch(depth_error), mask=squeeze_batch(mask) if mask is not None else None),
                    blank,
                ],
                axis=1,
            )
        )

    vis_path = output_dir / "vis" / f"{sample_name}.png"
    Image.fromarray(np.concatenate(rows, axis=0)).save(vis_path)

    paths = {
        "vis": vis_path,
        "depth": save_float_output(output_dir / "depth" / f"{sample_name}.exr", pred_depth_hw),
        "points": save_float_output(output_dir / "points" / f"{sample_name}.exr", pred_points, convert_rgb_to_bgr=True),
        "normal": save_float_output(
            output_dir / "normal" / f"{sample_name}.exr",
            squeeze_batch(pred_normal_map).transpose(1, 2, 0) * np.array([1, -1, -1], dtype=np.float32)
            if squeeze_batch(pred_normal_map).ndim == 3 and squeeze_batch(pred_normal_map).shape[0] == 3
            else squeeze_batch(pred_normal_map),
            convert_rgb_to_bgr=True,
            exr_type=cv2.IMWRITE_EXR_TYPE_HALF,
        ),
        "img": output_dir / "img" / f"{sample_name}.png",
    }
    Image.fromarray(rgb_uint8).save(paths["img"])

    if mask is not None:
        mask_path = output_dir / "mask" / f"{sample_name}.png"
        Image.fromarray(squeeze_batch(mask).astype(np.uint8) * 255).save(mask_path)
        paths["mask"] = mask_path

    return paths
