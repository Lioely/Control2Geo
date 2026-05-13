import numpy as np
import pandas as pd
from torch.utils.data import (
    Dataset, 
)
from data.dataset.utils import process_data,affine_invariant_point_map,read_data
import utils3d
from torchvision.transforms import (
    InterpolationMode, 
    Resize, 
)
def dist_2_depth(width, height, flt_focal, distance):
    img_plane_x = (
        np.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width)
        .reshape(1, width)
        .repeat(height, 0)
        .astype(np.float32)[:, :, None]
    )
    img_plane_y = (
        np.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height)
        .reshape(height, 1)
        .repeat(width, 1)
        .astype(np.float32)[:, :, None]
    )
    img_plane_z = np.full([height, width, 1], flt_focal, np.float32)
    img_plane = np.concatenate([img_plane_x, img_plane_y, img_plane_z], 2)

    depth = distance / np.linalg.norm(img_plane, 2, 2) * flt_focal
    return depth

class Hypersim_Dataset(Dataset):
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
        gt_path = self.filenames_pd['depth'][idx].replace("\\", "/")
        normal_path = self.filenames_pd['normal'][idx].replace("\\", "/")
        control_path = self.filenames_pd['control_geo'][idx].replace("\\", "/")
        
        image = read_data(data_path)
        image = image.astype(np.uint8) # type: ignore
        depth = read_data(gt_path,isdepth=True)

        depth = dist_2_depth(self.IMG_WIDTH,
                                self.IMG_HEIGHT,
                                self.focal[0],
                                depth)
        depth = depth
        depth_mask = np.logical_and((depth >= self.min_depth), (depth <= self.max_depth)).astype(np.bool_)

        normal = read_data(normal_path)
        # 将三个维度全为0的像素设置为[0,0,1]
        mask = (normal[..., 0] == 0) & (normal[..., 1] == 0) & (normal[..., 2] == 0)
        normal[mask] = [128, 128, 255]
        normal = normal.astype(np.uint8)

        control_map = read_data(control_path,isdepth=True)
        control_map = (control_map/65535.0)

        
        proc = process_data({
            'image': image,
            'depth': depth,
            'control_geo': control_map,
            'depth_mask': depth_mask,
            'normal': normal,
            "width":self.width,
            "height":self.height,
            'intrinsics': self.intrinsics,
            "depth_range": [self.min_depth, self.max_depth]
        }, self.dataset_args)
        #adjust depth max value
        #1212
        #affine_v_p has been normalized to [-1,1]
        pts = utils3d.pt.depth_map_to_point_map(proc['depth'], intrinsics=proc['intrinsics'])
        affine_p = affine_invariant_point_map(pts, mask=None)
        point_t = affine_p.permute(0,3,1,2).float()
        point_t = point_t.squeeze()
        proc["point"] = point_t        
        return proc

    def __len__(self):
        return len(self.filenames_pd)


if "__main__" == __name__:
    from omegaconf import OmegaConf
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from ....utils.vis import colorize_depth_affine,colorize_normal
    import trimesh
    config = OmegaConf.load("config/train_data.yaml")
    dataset = Hypersim_Dataset(**config["hypersim"])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
    for proc in dataloader:
        depth = proc['depth']
        image = proc['image']
        point = proc['point']
        print(proc['point'].shape,type(proc['point']))
        print(proc['depth'].shape,type(proc['depth']))
        print(proc['image'].shape,type(proc['image']))
        break
    depth1 = depth[0]
    image1 = image[0]
    point1 = point[0]
    # test rgb
    rgb = image1.permute(1,2,0)
    rgb = (((np.asarray(rgb)+1.0)/2.0)*255.0).astype(np.uint8)
    print(rgb.shape)
    Image.fromarray(rgb).save("rgb1.png")
    # test depth 
    color_depth = colorize_depth_affine(np.asarray(depth1.squeeze()))
    print(color_depth.shape)
    Image.fromarray(color_depth).save("depth1.png")
    # test normal
    affine_v_p = point1.permute(1,2,0)
    affine_v_p = np.asarray(affine_v_p)
    print(affine_v_p.shape)
    gt_depth = np.asarray(depth1.squeeze())
    mask = np.logical_and((gt_depth >= 1e-5), (gt_depth <= 65.0)).astype(np.bool_)
    normals1, normals_mask1 = utils3d.numpy.points_to_normals(affine_v_p, mask=mask)
    Image.fromarray(colorize_normal(normals1)).save("1.png")
    # test affine_v_p meash
    H,W = affine_v_p.shape[:2]
    faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
    affine_v_p,
    rgb.astype(np.float32) / 255,
    utils3d.numpy.image_uv(width=W, height=H),
    mask=mask & ~(utils3d.numpy.depth_edge(gt_depth, rtol=0.03, mask=mask) & utils3d.numpy.normals_edge(normals1, tol=5, mask=normals_mask1)),
    tri=True
    )
    vertices, vertex_uvs = vertices * np.array([1, -1, -1]), vertex_uvs * np.array([1, -1]) + np.array([0, 1])
    mesh = trimesh.Trimesh(
                    vertices=vertices,
                    vertex_colors=vertex_colors,
                    faces=faces, 
                    process=False
    )
    mesh.show()
