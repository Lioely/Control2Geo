import numpy as np
import pandas as pd
import torch
import utils3d
import json
from PIL import Image
import cv2
from torch.utils.data import (
    Dataset, 
)
from data.dataset.utils import (
    affine_invariant_point_map,
    read_depth,
    read_data
)
from utils.geometry_numpy import (
    focal_to_fov_numpy, norm3d
)


class Eval_Dataset(Dataset):
    def __init__(self, 
                path,
                width,
                height,
                depth_range,
                resize_shape,
                depth_unit = 1.0,
                drop_max_depth = 1000.,
                **kwargs):
        """
        path --> .csv file 
        width,height --> crop size/resize of images to test model 
        depth_unit --> metric depth unit, defalut '1.0'
        """
        self.filenames_pd = pd.read_csv(path)
        self.dataset_args = kwargs
        self.depth_unit   = depth_unit
        self.drop_max_depth = drop_max_depth
        self.test_width, self.test_height = resize_shape
        self.IMG_WIDTH,self.IMG_HEIGHT = width,height


    def __getitem__(self, idx):
        data_path = self.filenames_pd['image'][idx].replace("\\", "/")
        gt_path = self.filenames_pd['depth'][idx].replace("\\", "/")
        with open(self.filenames_pd['meta'][idx].replace("\\", "/"), 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        control_path = self.filenames_pd['control_geo'][idx].replace("\\", "/")
        intrinsics = np.asarray(meta_data["intrinsics"])
        image = read_data(data_path)
        image = image.astype(np.uint8) # type: ignore

        control_map = read_data(control_path, isdepth=True)
        control_map = (control_map / 65535.0).astype(np.float32)

        depth, _ = read_depth(gt_path)
        proc = {
            'image': image,
            'depth': np.nan_to_num(depth, nan=1, posinf=1, neginf=1),
            'depth_mask': np.isfinite(depth),
            'control_geo': control_map,
            'intrinsics': intrinsics,
            "width":self.IMG_WIDTH,
            "height":self.IMG_HEIGHT,
        }
        #affine_v_p has been normalized to [-1,1]
        proc = self.process_instance(proc)
        return proc
    
    def process_instance(self, instance: dict):
        if instance is None:
            return None
        
        image, depth, depth_mask, intrinsics = instance['image'], instance['depth'], instance['depth_mask'], instance['intrinsics']
        control_map =  instance['control_geo']


        raw_height, raw_width = image.shape[:2]
        raw_horizontal, raw_vertical = abs(1.0 / intrinsics[0, 0]), abs(1.0 / intrinsics[1, 1])
        raw_pixel_w, raw_pixel_h = raw_horizontal / raw_width, raw_vertical / raw_height
        tgt_width, tgt_height = instance['width'], instance['height']
        tgt_aspect = tgt_width / tgt_height

        # set expected target view field
        tgt_horizontal = min(raw_horizontal, raw_vertical * tgt_aspect)
        tgt_vertical = tgt_horizontal / tgt_aspect

        # set target view direction
        cu, cv = 0.5, 0.5
        direction = utils3d.np.unproject_cv(np.array([[cu, cv]], dtype=np.float32), np.array([1.0], dtype=np.float32), intrinsics=intrinsics)[0]
        R = utils3d.np.rotation_matrix_from_vectors(direction, np.array([0, 0, 1], dtype=np.float32))

        # restrict target view field within the raw view
        corners = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
        corners = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1) @ (np.linalg.inv(intrinsics).T @ R.T)   # corners in viewport's camera plane
        corners = corners[:, :2] / corners[:, 2:3]

        warp_horizontal, warp_vertical = abs(1.0 / intrinsics[0, 0]), abs(1.0 / intrinsics[1, 1])
        for i in range(4):
            intersection, _ = utils3d.np.ray_intersection(
                np.array([0., 0.]), np.array([[tgt_aspect, 1.0], [tgt_aspect, -1.0]]),
                corners[i - 1], corners[i] - corners[i - 1],
            )
            warp_horizontal, warp_vertical = min(warp_horizontal, 2 * np.abs(intersection[:, 0]).min()), min(warp_vertical, 2 * np.abs(intersection[:, 1]).min())
        tgt_horizontal, tgt_vertical = min(tgt_horizontal, warp_horizontal), min(tgt_vertical, warp_vertical)

        # get target view intrinsics
        fx, fy = 1.0 / tgt_horizontal, 1.0 / tgt_vertical
        tgt_intrinsics = utils3d.np.intrinsics_from_focal_center(fx, fy, 0.5, 0.5).astype(np.float32)
        
        # do homogeneous transformation with the rotation and intrinsics
        # 4.1 The image and depth is resized first to approximately the same pixel size as the target image with PIL's antialiasing resampling
        tgt_pixel_w, tgt_pixel_h = tgt_horizontal / tgt_width, tgt_vertical / tgt_height        # (should be exactly the same for x and y axes)
        rescaled_w, rescaled_h = int(raw_width * raw_pixel_w / tgt_pixel_w), int(raw_height * raw_pixel_h / tgt_pixel_h)
        image = np.array(Image.fromarray(image).resize((rescaled_w, rescaled_h), Image.Resampling.LANCZOS))
        control_map = cv2.resize(control_map, (rescaled_w, rescaled_h), interpolation=cv2.INTER_LANCZOS4)

        depth, depth_mask = utils3d.np.masked_nearest_resize(depth, mask=depth_mask, size=(rescaled_h, rescaled_w))
        distance = norm3d(utils3d.np.depth_map_to_point_map(depth, intrinsics=intrinsics))
        # 4.2 calculate homography warping
        transform = intrinsics @ np.linalg.inv(R) @ np.linalg.inv(tgt_intrinsics)
        uv_tgt = utils3d.np.uv_map(tgt_height, tgt_width)
        pts = np.concatenate([uv_tgt, np.ones((tgt_height, tgt_width, 1), dtype=np.float32)], axis=-1) @ transform.T
        uv_remap = pts[:, :, :2] / (pts[:, :, 2:3] + 1e-12)
        pixel_remap = utils3d.np.uv_to_pixel(uv_remap, (rescaled_h, rescaled_w)).astype(np.float32)
        
        tgt_image = cv2.remap(image, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_LINEAR)
        tgt_control = cv2.remap(control_map, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_LINEAR)
        tgt_distance = cv2.remap(distance, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST)
        tgt_ray_length = utils3d.np.unproject_cv(uv_tgt, np.ones_like(uv_tgt[:, :, 0]), intrinsics=tgt_intrinsics)
        tgt_ray_length = (tgt_ray_length[:, :, 0] ** 2 + tgt_ray_length[:, :, 1] ** 2 + tgt_ray_length[:, :, 2] ** 2) ** 0.5
        tgt_depth = tgt_distance / (tgt_ray_length + 1e-12)
        tgt_depth_mask = cv2.remap(depth_mask.astype(np.uint8), pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST) > 0
    
        # drop depth greater than drop_max_depth
        max_depth = np.nanquantile(np.where(tgt_depth_mask, tgt_depth, np.nan), 0.01) * self.drop_max_depth
        tgt_depth_mask &= tgt_depth <= max_depth
        tgt_depth = np.nan_to_num(tgt_depth, nan=0.0)

        if self.depth_unit is not None:
            tgt_depth *= self.depth_unit
        
        if not np.any(tgt_depth_mask):
            # always make sure that mask is not empty, otherwise the loss calculation will crash
            tgt_depth_mask = np.ones_like(tgt_depth_mask)
            tgt_depth = np.ones_like(tgt_depth)
            instance['label_type'] = 'invalid'
            
        test_image = np.array(Image.fromarray(tgt_image).resize((self.test_width, self.test_height), Image.Resampling.LANCZOS)) 
        tgt_control = cv2.resize(tgt_control, (self.test_width, self.test_height), interpolation=cv2.INTER_LANCZOS4)
        
        tgt_image = (tgt_image.astype(np.float32) / 255.0) * 2.0 - 1.0
        tgt_image = np.nan_to_num(tgt_image, nan=1.0, posinf=1.0, neginf=-1.0)
        tgt_image = np.clip(tgt_image, -1.0, 1.0)

        
        test_image = (test_image.astype(np.float32) / 255.0) * 2.0 - 1.0
        test_image = np.nan_to_num(test_image, nan=1.0, posinf=1.0, neginf=-1.0)
        test_image = np.clip(test_image, -1.0, 1.0)
        
        tgt_control = tgt_control.astype(np.float32) * 2.0 - 1.0
        tgt_control = np.clip(tgt_control, -1.0, 1.0)
        
        tgt_pts = utils3d.np.unproject_cv(uv_tgt, tgt_depth, intrinsics=tgt_intrinsics)
        #tgt_pts = affine_invariant_point_map(tgt_pts, tgt_depth_mask)

        instance.update({
            'image': torch.from_numpy(test_image.astype(np.float32)).permute(2, 0, 1),
            'ori_image':torch.from_numpy(tgt_image).permute(2, 0, 1),
            'depth': torch.from_numpy(tgt_depth).float(),
            'control_geo':torch.from_numpy(tgt_control).unsqueeze(0).float(),
            'depth_mask': torch.from_numpy(tgt_depth_mask).bool(),
            'intrinsics': torch.from_numpy(tgt_intrinsics).float(),
            'point': torch.from_numpy(tgt_pts).permute(2, 0, 1).float(),
        })
                
        return instance

    def __len__(self):
        return len(self.filenames_pd)