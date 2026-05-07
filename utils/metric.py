from typing import *
from numbers import Number
import torch.nn.functional as F
import torch
import utils3d
from utils.alignment import (
    align_points_scale_z_shift, 
    align_points_scale_xyz_shift, 
    align_points_xyz_shift,
    align_affine_lstsq, 
    align_depth_scale, 
    align_depth_affine, 
    align_points_scale,
)
from utils.geometry_torch import (
    weighted_mean, 
    intrinsics_to_fov
)

def rel_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    
    rel = (torch.abs(pred - gt) / (gt + eps)).mean()
    #print("rel_depth:",rel)
    return rel.item()


def delta1_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    delta1 = (torch.maximum(gt / pred, pred / gt) < 1.25).float().mean()
    return delta1.item()


def rel_point(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    dist_gt = torch.norm(gt, dim=-1)
    dist_err = torch.norm(pred - gt, dim=-1)
    rel = (dist_err / (dist_gt + eps)).mean()
    return rel.item()


def delta1_point(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    dist_pred = torch.norm(pred, dim=-1)
    dist_gt = torch.norm(gt, dim=-1)
    dist_err = torch.norm(pred - gt, dim=-1)

    delta1 = (dist_err < 0.25 * torch.minimum(dist_gt, dist_pred)).float().mean()
    return delta1.item()


def rel_point_local(pred: torch.Tensor, gt: torch.Tensor, diameter: torch.Tensor):
    dist_err = torch.norm(pred - gt, dim=-1)
    rel = (dist_err / diameter).mean()
    return rel.item()


def delta1_point_local(pred: torch.Tensor, gt: torch.Tensor, diameter: torch.Tensor):
    dist_err = torch.norm(pred - gt, dim=-1)
    delta1 = (dist_err < 0.25 * diameter).float().mean()
    return delta1.item()


def boundary_f1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, radius: int = 1):
    neighbor_x, neight_y = torch.meshgrid(
        torch.linspace(-radius, radius, 2 * radius + 1, device=pred.device),
        torch.linspace(-radius, radius, 2 * radius + 1, device=pred.device),
        indexing='xy'
    )
    neighbor_mask = (neighbor_x ** 2 + neight_y ** 2) <= radius ** 2 + 1e-5

    pred_window = utils3d.torch.sliding_window_2d(pred, window_size=2 * radius + 1, stride=1, dim=(-2, -1))                 # [H, W, 2*R+1, 2*R+1]
    gt_window = utils3d.torch.sliding_window_2d(gt, window_size=2 * radius + 1, stride=1, dim=(-2, -1))                     # [H, W, 2*R+1, 2*R+1]
    mask_window = neighbor_mask & utils3d.torch.sliding_window_2d(mask, window_size=2 * radius + 1, stride=1, dim=(-2, -1)) # [H, W, 2*R+1, 2*R+1]

    pred_rel = pred_window / pred[radius:-radius, radius:-radius, None, None]
    gt_rel = gt_window / gt[radius:-radius, radius:-radius, None, None]
    valid = mask[radius:-radius, radius:-radius, None, None] & mask_window
    
    f1_list = []
    w_list = t_list = torch.linspace(0.05, 0.25, 10).tolist()

    for t in t_list:
        pred_label = pred_rel > 1 + t
        gt_label = gt_rel > 1 + t
        TP = (pred_label & gt_label & valid).float().sum()
        precision = TP / (gt_label & valid).float().sum().clamp_min(1e-12)
        recall = TP / (pred_label & valid).float().sum().clamp_min(1e-12)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        f1_list.append(f1.item())
    
    f1_avg = sum(w * f1 for w, f1 in zip(w_list, f1_list)) / sum(w_list)
    return f1_avg


def compute_metrics(
    pred: Dict[str, torch.Tensor], 
    gt: Dict[str, torch.Tensor], 
) -> Tuple[Dict[str, Dict[str, Number]], Dict[str, torch.Tensor]]:

    metrics = {}
    #[B,C,H,W]
    mask = gt['depth_mask'].detach().cpu()
    gt_depth = gt['depth'].detach().cpu()
    gt_points = gt['point'].detach().cpu()
    height, width = mask.shape[-2:]
    lr_mask, lr_index = utils3d.pt.masked_nearest_resize(mask=mask, size=(64, 64), return_index=True)
    
    pred_points_affine_invariant = pred['point'].detach().cpu()
    if pred_points_affine_invariant.shape[-2] != height or pred_points_affine_invariant.shape[-1] != width:
        pred_points_affine_invariant = F.interpolate(pred_points_affine_invariant, size=(height, width), mode='bilinear')
    pred_points_affine_invariant = pred_points_affine_invariant.detach().cpu()
    pred_points_affine_invariant = pred_points_affine_invariant.permute(0,2,3,1).detach().cpu()
    gt_points = gt_points.permute(0,2,3,1)
    pred_points_lr_masked, gt_points_lr_masked = pred_points_affine_invariant[lr_index][lr_mask], gt_points[lr_index][lr_mask]
    #print(pred_points_lr_masked.shape, gt_points_lr_masked.shape,(1 / gt_points_lr_masked.norm(dim=-1)).shape)
    
    scale, shift = align_points_scale_xyz_shift(pred_points_lr_masked[None,...], gt_points_lr_masked[None,...], (1 / gt_points_lr_masked.norm(dim=-1))[None,...])
    pred_points_aligned = pred_points_affine_invariant * scale + shift
    
    metrics['points_affine_invariant'] = {
        'rel': rel_point(pred_points_aligned[mask], gt_points[mask]),
        'delta1': delta1_point(pred_points_aligned[mask], gt_points[mask])
    }
        
    pred_depth_affine_invariant = pred['depth'].unsqueeze(1)
    if pred_depth_affine_invariant.shape[-2] != height or pred_depth_affine_invariant.shape[-1] != width:
        pred_depth_affine_invariant = F.interpolate(pred_depth_affine_invariant, size=(height, width), mode='bilinear')
        
    pred_depth_affine_invariant = pred_depth_affine_invariant.squeeze(1).detach().cpu()
    pred_depth_lr_masked, gt_depth_lr_masked = pred_depth_affine_invariant[lr_index][lr_mask], gt_depth[lr_index][lr_mask]
    scale, shift = align_depth_affine(pred_depth_lr_masked, gt_depth_lr_masked, 1 / gt_depth_lr_masked)
    pred_depth_aligned = pred_depth_affine_invariant * scale + shift

    metrics['depth_affine_invariant'] = {
        'rel': rel_depth(pred_depth_aligned[mask], gt_depth[mask]),
        'delta1': delta1_depth(pred_depth_aligned[mask], gt_depth[mask])
    }
    #print(torch.isfinite(pred_points_aligned[mask]).any(),torch.isfinite(pred_depth_aligned[mask]).any(),torch.isfinite(gt_points[mask]).any(),torch.isfinite(gt_depth[mask]).any())
    return metrics, pred_points_aligned, pred_depth_aligned

def compute_metrics_for_logger(image):
    metrics = {}
    
    gt_depth = image['gt_depth'].detach().cpu()
    gt_points = image['gt_point_map'].detach().cpu()
    mask = torch.ones_like(gt_depth).bool()
    height, width = mask.shape[-2:]
    lr_mask, lr_index = utils3d.pt.masked_nearest_resize(mask=mask, size=(64, 64), return_index=True)
    #bhwc
    pred_points_affine_invariant = image['pred_point_map'].detach().cpu()
    #gt_points = gt_points
    pred_points_lr_masked, gt_points_lr_masked = pred_points_affine_invariant[lr_index][lr_mask], gt_points[lr_index][lr_mask]
    #print(pred_points_lr_masked.shape, gt_points_lr_masked.shape,(1 / gt_points_lr_masked.norm(dim=-1)).shape)
    
    scale, shift = align_points_scale_xyz_shift(pred_points_lr_masked[None,...], gt_points_lr_masked[None,...], (1 / gt_points_lr_masked.norm(dim=-1))[None,...])
    pred_points_aligned = pred_points_affine_invariant * scale + shift

    metrics['points_affine_invariant'] = {
        'rel': rel_point(pred_points_aligned[mask], gt_points[mask]),
        'delta1': delta1_point(pred_points_aligned[mask], gt_points[mask])
    }

    pred_depth_affine_invariant = image['pred_depth'].detach().cpu()
    pred_depth_lr_masked, gt_depth_lr_masked = pred_depth_affine_invariant[lr_index][lr_mask], gt_depth[lr_index][lr_mask]
    scale, shift = align_depth_affine(pred_depth_lr_masked, gt_depth_lr_masked, 1 / gt_depth_lr_masked)
    pred_depth_aligned = pred_depth_affine_invariant * scale + shift

    metrics['depth_affine_invariant'] = {
        'rel': rel_depth(pred_depth_aligned[mask], gt_depth[mask]),
        'delta1': delta1_depth(pred_depth_aligned[mask], gt_depth[mask])
    }

    return metrics