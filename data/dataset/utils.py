"""
adjusted from 
https://github.com/microsoft/MoGe/blob/main/moge/train/dataloader.py
"""
from pathlib import Path
import struct
import h5py
import scipy.io as scio
import io
from typing import *
import numpy as np
import utils3d
import cv2
from PIL import Image
import torchvision.transforms.v2.functional as TF
import torch
import os
import OpenEXR
import Imath
import imageio
import glob
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from utils.geometry_numpy import harmonic_mean_numpy, norm3d, depth_occlusion_edge_numpy
from utils.data_augmentation import sample_perspective, warp_perspective, image_color_augmentation


def read_pfm(filename):
    with Path(filename).open('rb') as pfm_file:
        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')

        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)

        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4

        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.flipud(np.reshape(decoded, shape)) * scale


def read_data(data_path,isdepth = False):
    suffix = data_path.split('.')[-1]
    if suffix == 'jpg' or suffix == 'png' or suffix == 'JPG':
        if isdepth:
            try:
                #data = Image.open(data_path)
                data = cv2.imread(data_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            except:
                print("error file")
                print(data_path)
        else:
            try:
                data = Image.open(data_path).convert('RGB')
            except:
                print("error file")
                print(data_path)
        data = np.asarray(data, dtype=np.float32)
    if suffix == 'hdf5':
        try:
            data = h5py.File(data_path, 'r')
            data = np.asarray(data['dataset'], dtype=np.float32)
        except:
            print("error file")
            print(data_path)
    if suffix == "exr":
        try:
            data = cv2.imread(data_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            data = np.asarray(data, dtype=np.float32)
        except:
            print("error file")
            print(data_path)
    if suffix =="npy":
        try:
            data = np.load(data_path).squeeze()
        except:
            print("error file")
            print(data_path)
    return data

def read_depth(path: Union[str, os.PathLike, IO]) -> Tuple[np.ndarray, float]:
    """
    Read a depth image, return float32 depth array of shape (H, W).
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    pil_image = Image.open(io.BytesIO(data))
    near = float(pil_image.info.get('near'))
    far = float(pil_image.info.get('far'))
    unit = float(pil_image.info.get('unit')) if 'unit' in pil_image.info else None
    depth = np.array(pil_image)
    mask_nan, mask_inf = depth == 0, depth == 65535
    depth = (depth.astype(np.float32) - 1) / 65533
    depth = near ** (1 - depth) * far ** depth
    depth[mask_nan] = np.nan
    depth[mask_inf] = np.inf
    #valid_mask = ~np.logical_or(mask_nan,mask_inf)
    return depth, unit



def exr2hdr(exrpath):
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    CNum = len(File.header()['channels'].keys())
    if (CNum > 1):
    	Channels = ['R', 'G', 'B']
    	CNum = 3
    else:
    	Channels = ['G']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    Pixels = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in Channels]
    hdr = np.zeros((Size[1],Size[0],CNum),dtype=np.float32)
    if (CNum == 1):
        hdr[:,:,0] = np.reshape(Pixels[0],(Size[1],Size[0]))
    else:
	    hdr[:,:,0] = np.reshape(Pixels[0],(Size[1],Size[0]))
	    hdr[:,:,1] = np.reshape(Pixels[1],(Size[1],Size[0]))
	    hdr[:,:,2] = np.reshape(Pixels[2],(Size[1],Size[0]))
    return hdr

def writehdr(hdrpath,hdr):
	h, w, c = hdr.shape
	if c == 1:
		hdr = np.pad(hdr, ((0, 0), (0, 0), (0, 2)), 'constant')
		hdr[:,:,1] = hdr[:,:,0]
		hdr[:,:,2] = hdr[:,:,0]
	imageio.imwrite(hdrpath,hdr,format='hdr')

def load_exr(filename):
	hdr = exr2hdr(filename)
	h, w, c = hdr.shape
	if c == 1:
		hdr = np.squeeze(hdr)
	return hdr
"""
def process_data(instance,dataset_args):
    raw_image, raw_depth, raw_normal, raw_intrinsics= \
            instance['image'], instance['depth'], \
            instance['normal'], instance['intrinsics']
    raw_control =  instance['control_geo']
    min_depth,max_depth = instance["depth_range"]
    
    center_augmentation = dataset_args["center_augmentation"]
    fov_range_absolute_min, fov_range_absolute_max = dataset_args['fov_range_absolute']
    fov_range_relative_min, fov_range_relative_max = dataset_args['fov_range_relative']
    image_augmentation = dataset_args['image_augmentation']

    raw_height, raw_width = raw_image.shape[:2]
    raw_fov_x, raw_fov_y = utils3d.np.intrinsics_to_fov(raw_intrinsics)
    tgt_width, tgt_height = instance['width'], instance['height']
    tgt_aspect = tgt_width / tgt_height
    
    rng = np.random.default_rng(dataset_args['seed'])
    # Sample perspective transformation
    tgt_intrinsics, R = sample_perspective(
        raw_intrinsics, 
        tgt_aspect=tgt_aspect,
        center_augmentation=center_augmentation,
        fov_range_absolute=dataset_args['fov_range_absolute'],
        fov_range_relative=dataset_args['fov_range_relative'],
        rng=rng
    )

    # Warp
    transform = tgt_intrinsics @ R @ np.linalg.inv(raw_intrinsics)
    # - Warp image
    tgt_image = warp_perspective(raw_image, transform, tgt_size=(tgt_height, tgt_width), interpolation='lanczos')
    tgt_control = warp_perspective(raw_control, transform, tgt_size=(tgt_height, tgt_width), interpolation='lanczos')
    tgt_normal = warp_perspective(raw_normal, transform, tgt_size=(tgt_height, tgt_width), interpolation='lanczos')
    # - Warp depth
    depth_edge_mask = utils3d.np.depth_map_edge(raw_depth, mask=np.isfinite(raw_depth), kernel_size=5, ltol=0.01)
    depth_bilinear_mask = np.isfinite(raw_depth) & ~depth_edge_mask 
    warped_depth_bilinear_mask = warp_perspective(depth_bilinear_mask.astype(np.float32), transform, (tgt_height, tgt_width), interpolation='bilinear')
    warped_depth_nearest = warp_perspective(raw_depth, transform, (tgt_height, tgt_width), interpolation='nearest', sparse_mask=~np.isnan(raw_depth))
    warped_depth_bilinear = 1 / warp_perspective(1 / raw_depth, transform, (tgt_height, tgt_width), interpolation='bilinear')   # NOTE: Bilinear intepolation in disparity space maintains planar surfaces.
    warped_depth = np.where(warped_depth_bilinear_mask == 1., warped_depth_bilinear, warped_depth_nearest)
    tgt_uvhomo = np.concatenate([utils3d.np.uv_map((tgt_height, tgt_width)), np.ones((tgt_height, tgt_width, 1), dtype=np.float32)], axis=-1)
    tgt_depth = warped_depth / np.dot(tgt_uvhomo, np.linalg.inv(transform)[2, :])

     #always make sure that mask is not empty
    if np.isfinite(tgt_depth).sum() / tgt_depth.size < 0.001:
        tgt_depth = np.ones_like(tgt_depth)
        instance['label_type'] = 'invalid'

    # Flip augmentation
    if rng.choice([True, False]):
        tgt_image  = np.flip(tgt_image, axis=1).copy()
        tgt_normal = np.flip(tgt_normal, axis=1).copy()
        tgt_depth = np.flip(tgt_depth, axis=1).copy()
        tgt_control = np.flip(tgt_control, axis=1).copy()
        tgt_depth_mask = np.flip(tgt_depth_mask, axis=1).copy()
        #tgt_depth_mask_inf = np.flip(tgt_depth_mask_inf, axis=1).copy()

    tgt_image = image_color_augmentation(
            tgt_image, 
            augmentations=image_augmentation, 
            rng=rng, 
            depth=tgt_depth,
        )

    depth_mask = np.logical_and((tgt_depth >= min_depth), (tgt_depth <= max_depth)).astype(np.bool_)
    tgt_depth = np.clip(tgt_depth, np.percentile(tgt_depth[depth_mask], 2), np.percentile(tgt_depth[depth_mask], 98))
    
    to_tensor = lambda x, dtype=None: torch.as_tensor(x, dtype=dtype)
    # normalized image to [-1,1]
    tgt_image = (tgt_image.astype(np.float32) / 255.0) * 2.0 - 1.0
    tgt_image = np.nan_to_num(tgt_image, nan=1.0, posinf=1.0, neginf=-1.0)
    tgt_image = np.clip(tgt_image, -1.0, 1.0)

    tgt_normal = (tgt_normal.astype(np.float32) / 255.0) * 2.0 - 1.0
    tgt_normal = np.clip(tgt_normal, -1.0, 1.0)

    tgt_control = (tgt_control.astype(np.float32) / 255.0)
    instance.update({
            'image': to_tensor(tgt_image).permute(2, 0, 1).float(),
            'depth': to_tensor(tgt_depth).unsqueeze(0).float(),
            'normal':to_tensor(tgt_normal).permute(2, 0, 1).float(),
            'control_geo':to_tensor(tgt_control).unsqueeze(0).float(),
            'intrinsics': to_tensor(tgt_intrinsics).float()
        })
        
    return instance
"""  

def process_data(instance,dataset_args):
    raw_image, raw_depth, raw_normal, raw_intrinsics= \
            instance['image'], instance['depth'], \
            instance['normal'], instance['intrinsics']
    raw_control =  instance['control_geo']
    min_depth,max_depth = instance["depth_range"]
    
    center_augmentation = dataset_args["center_augmentation"]
    fov_range_absolute_min, fov_range_absolute_max = dataset_args['fov_range_absolute']
    fov_range_relative_min, fov_range_relative_max = dataset_args['fov_range_relative']
    image_augmentation = dataset_args['image_augmentation']

    raw_height, raw_width = raw_image.shape[:2]
    raw_fov_x, raw_fov_y = utils3d.np.intrinsics_to_fov(raw_intrinsics)
    tgt_width, tgt_height = instance['width'], instance['height']
    tgt_aspect = tgt_width / tgt_height
    
    rng = np.random.default_rng(dataset_args['seed'])
    # Sample perspective transformation
    tgt_intrinsics, R = sample_perspective(
        raw_intrinsics, 
        tgt_aspect=tgt_aspect,
        center_augmentation=center_augmentation,
        fov_range_absolute=dataset_args['fov_range_absolute'],
        fov_range_relative=dataset_args['fov_range_relative'],
        rng=rng
    )

    # Warp
    transform = tgt_intrinsics @ R @ np.linalg.inv(raw_intrinsics)
    # - Warp image
    tgt_image = warp_perspective(raw_image, transform, tgt_size=(tgt_height, tgt_width), interpolation='lanczos')
    tgt_control = warp_perspective(raw_control, transform, tgt_size=(tgt_height, tgt_width), interpolation='lanczos')
    tgt_normal = warp_perspective(raw_normal, transform, tgt_size=(tgt_height, tgt_width), interpolation='lanczos')
    # - Warp depth
    depth_edge_mask = utils3d.np.depth_map_edge(raw_depth, mask=np.isfinite(raw_depth), kernel_size=5, ltol=0.01)
    depth_bilinear_mask = np.isfinite(raw_depth) & ~depth_edge_mask 
    warped_depth_bilinear_mask = warp_perspective(depth_bilinear_mask.astype(np.float32), transform, (tgt_height, tgt_width), interpolation='bilinear')
    warped_depth_nearest = warp_perspective(raw_depth, transform, (tgt_height, tgt_width), interpolation='nearest', sparse_mask=~np.isnan(raw_depth))
    warped_depth_bilinear = 1 / warp_perspective(1 / raw_depth, transform, (tgt_height, tgt_width), interpolation='bilinear')   # NOTE: Bilinear intepolation in disparity space maintains planar surfaces.
    warped_depth = np.where(warped_depth_bilinear_mask == 1., warped_depth_bilinear, warped_depth_nearest)
    tgt_uvhomo = np.concatenate([utils3d.np.uv_map((tgt_height, tgt_width)), np.ones((tgt_height, tgt_width, 1), dtype=np.float32)], axis=-1)
    tgt_depth = warped_depth / np.dot(tgt_uvhomo, np.linalg.inv(transform)[2, :])

    tgt_depth_mask = np.isfinite(tgt_depth)

    #always make sure that mask is not empty
    if tgt_depth_mask.sum() / tgt_depth.size < 0.001:
        tgt_depth = np.ones_like(tgt_depth)
        tgt_depth_mask = np.ones_like(tgt_depth_mask)
        instance['label_type'] = 'invalid'

    # Flip augmentation
    if rng.choice([True, False]):
        tgt_image  = np.flip(tgt_image, axis=1).copy()
        tgt_normal = np.flip(tgt_normal, axis=1).copy()
        tgt_depth = np.flip(tgt_depth, axis=1).copy()
        tgt_control = np.flip(tgt_control, axis=1).copy()
        tgt_depth_mask = np.flip(tgt_depth_mask, axis=1).copy()

    tgt_image = image_color_augmentation(
            tgt_image, 
            augmentations=image_augmentation, 
            rng=rng, 
            depth=tgt_depth,
        )

    depth_mask = np.logical_and((tgt_depth >= min_depth), (tgt_depth <= max_depth)).astype(np.bool_)
    tgt_depth = np.clip(tgt_depth, min_depth, np.percentile(tgt_depth[depth_mask], 98))
    
    to_tensor = lambda x, dtype=None: torch.as_tensor(x, dtype=dtype)
    # normalized image to [-1,1]
    tgt_image = (tgt_image.astype(np.float32) / 255.0) * 2.0 - 1.0
    tgt_image = np.nan_to_num(tgt_image, nan=1.0, posinf=1.0, neginf=-1.0)
    tgt_image = np.clip(tgt_image, -1.0, 1.0)

    tgt_normal = (tgt_normal.astype(np.float32) / 255.0) * 2.0 - 1.0
    tgt_normal = np.clip(tgt_normal, -1.0, 1.0)

    tgt_control = tgt_control.astype(np.float32) * 2.0 - 1.0
    tgt_control = np.clip(tgt_control, -1.0, 1.0)
    instance.update({
            'image': to_tensor(tgt_image).permute(2, 0, 1).float(),
            'depth': to_tensor(tgt_depth).unsqueeze(0).float(),
            'normal':to_tensor(tgt_normal).permute(2, 0, 1).float(),
            'control_geo':to_tensor(tgt_control).unsqueeze(0).float(),
            'intrinsics': to_tensor(tgt_intrinsics).float()
        })
        
    return instance

def affine_invariant_point_map(point_map: torch.Tensor,
                               mask: torch.Tensor = None) -> torch.Tensor:
    """
    Args:
        point_map: [B, H, W, 3] float tensor
        mask:      [B, H, W]    bool tensor, True 表示有效点；若为 None 则全点有效
    Returns:
        [B, H, W, 3] 经过仿射不变归一化后的点图
    """
    B, H, W, C = point_map.shape
    assert C == 3, "last dim must be 3"
    # 展平成 [B, N, 3]
    N = H * W
    pts = point_map.view(B, N, 3)
    # 构造 mask_f: [B, N, 1]，无 mask 则全 1
    if mask is not None:
        #real/sparse data for training
        m = mask.view(B, N).unsqueeze(-1)    # bool
        mask_f = m.to(point_map.dtype)       # float 0/1
    else:
        #dense input for training
        mask_f = torch.ones((B, N, 1), device=pts.device, dtype=pts.dtype)

    # 1) 计算每 batch 的有效点数量 [B,1]
    counts = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)

    # 2) 计算中心 center [B,1,3]
    center = (pts * mask_f).sum(dim=1, keepdim=True) / counts

    # 3) 去中心化并算 scale
    centered = (pts - center) * mask_f       # [B, N, 3]，无效点为 0
    mse = (centered.pow(2).sum(dim=2)).sum(dim=1, keepdim=True) / counts.squeeze(-1)  # [B,1]
    scale = torch.sqrt(mse).unsqueeze(-1)    # [B,1,1]

    # 4) 归一化有效点
    normalized = (pts - center) / (scale + 1e-8)  # [B, N, 3]

    # 5) 构造新的 pts：有效点用 normalized，无效点保持原值
    new_pts = torch.where(mask_f.bool(), normalized, pts)
    #print("masked:",new_pts[mask_f.bool()].shape)
   # 6) 只对 mask 内点求绝对值最大
    #    a) 先把无效点设为 -inf，这样它们不会影响 max
    abs_vals = new_pts.abs().masked_fill(~mask_f.bool(), float("-inf"))  # [B, N, 3]
    #    b) 沿 N,C 维度取最大，得到 [B,1,1]
    abs_max = abs_vals.amax(dim=(1,2), keepdim=True).clamp(min=1e-8)      # [B,1,1]

    # 7) 用 abs_max 缩放所有点
    new_pts = new_pts / abs_max                                           # [B, N, 3]
    # 8) 把 mask 外的位置全部设为 1
    new_pts = torch.where(mask_f.bool(), new_pts, torch.tensor(1.0, device=new_pts.device, dtype=new_pts.dtype))
    new_pts= new_pts.clamp(-1,1)
    # 9) 恢复形状并返回
    return new_pts.view(B, H, W, 3)
"""
def affine_invariant_point_map(point_map, mask=None):
    
    #point_map: [H, W, 3]
    #mask: [H, W] optional, bool tensor, where True indicates valid points
    #Returns: affine-invariant point map [H, W, 3]
    
    H, W, _ = point_map.shape
    points = point_map.reshape(-1, 3)  # [H*W, 3]
    
    if mask is not None:
        mask_flat = mask.reshape(-1)  # [H*W]
        valid_points = points[mask_flat]
    else:
        valid_points = points
    
    # Compute center and RMS scale
    center = valid_points.mean(dim=0, keepdim=True)  # [1, 3]
    centered = valid_points - center                 # [N, 3]
    scale = torch.sqrt((centered ** 2).sum(dim=1).mean())  # scalar
    normalized = (valid_points - center) / scale     # [N, 3]

    # Replace original values
    new_points = points.clone()
    if mask is not None:
        new_points[mask_flat] = normalized
    else:
        new_points = normalized
    abs_value = max(abs(new_points.max()),abs(new_points.min()))
    new_points = new_points/abs_value
    return new_points.reshape(H, W, 3)
"""
