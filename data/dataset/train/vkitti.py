import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    Dataset, 
)
from data.dataset.utils import process_data,affine_invariant_point_map,read_data
import utils3d
from torchvision.transforms import (
    InterpolationMode, 
    Resize, 
)
#from ..utils import kitti_benchmark_crop,scale_norm
class VKitti_Dataset(Dataset):
    def __init__(self, 
                 ori_size,
                 path,
                 focal,
                 depth_range,
                 **kwargs):
        self.IMG_WIDTH,self.IMG_HEIGHT = ori_size
        self.focal = focal
        self.intrinsics = np.array([[focal[0]/self.IMG_WIDTH,    0, 0.5],
                          [   0, focal[1]/self.IMG_HEIGHT, 0.5], 
                          [   0,    0,    1]])
        self.filenames_pd = pd.read_csv(path)
        self.dataset_args = kwargs
        self.depth_scale, self.min_depth, self.max_depth = depth_range
        self.width, self.height = kwargs["tgt_size"][0],kwargs["tgt_size"][1]

    def __getitem__(self, idx):
        data_path = self.filenames_pd['image'][idx].replace("\\", "/")
        depth_path = self.filenames_pd['depth'][idx].replace("\\", "/")
        normal_path = self.filenames_pd['normal'][idx].replace("\\", "/")
        control_path = self.filenames_pd['control_geo'][idx].replace("\\", "/")
        
        image = read_data(data_path)
        image = image.astype(np.uint8) # type: ignore
        depth = read_data(depth_path,isdepth=True)

        depth = depth/100.0
        depth_mask = np.logical_and((depth >= self.min_depth), (depth <= self.max_depth)).astype(np.bool_)
        #depth  =  np.clip(depth, self.min_depth, self.max_depth)
        
        normal = read_data(normal_path)
        mask = (normal[..., 0] == 0) & (normal[..., 1] == 0) & (normal[..., 2] == 0)
        normal[mask] = [128, 128, 255]
        normal = normal.astype(np.uint8)

        control_map = read_data(control_path,isdepth=True)
        control_map = (control_map/65535.0)

        proc = process_data({
            'image': image,
            'depth': depth,
            'control_geo': control_map,
            'normal': normal,
            "width":self.width,
            'depth_mask': depth_mask,
            "height":self.height,
            'intrinsics': self.intrinsics,
            "depth_range": [self.min_depth, self.max_depth]
        }, self.dataset_args)
        #adjust depth max value
        #proc['depth'] = np.clip(proc['depth'], self.min_depth, self.max_depth)

        #affine_v_p has been normalized to [-1,1]
        pts = utils3d.pt.depth_map_to_point_map(proc['depth'], intrinsics=proc['intrinsics'])
        affine_p = affine_invariant_point_map(pts, mask=None)
        point_t = affine_p.permute(0,3,1,2).float()
        point_t = point_t.squeeze()
        proc["point"] = point_t        
        return proc

    def __len__(self):
        return len(self.filenames_pd)
