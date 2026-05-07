"""
https://github.com/microsoft/MoGe/blob/main/moge/utils/vis.py
we use the utils3d toolkit, so the camera intrinsics must be normalized
e.g.
    np.array([[fx/W,    0, 0.5], 
            [   0,fy/H, 0.5], 
            [   0,    0,    1]])
"""

import utils3d
import numpy as np
import trimesh
import matplotlib

def colorize_normal(normal: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    if mask is not None:
        normal = np.where(mask[..., None], normal, 0)
    normal = normal * [0.5, -0.5, -0.5] + 0.5
    normal = (normal.clip(0, 1) * 255).astype(np.uint8)
    return normal

def colorize_error_map(error_map: np.ndarray, mask: np.ndarray = None, cmap: str = 'plasma', value_range: tuple[float, float] = None):
    vmin, vmax = value_range if value_range is not None else (np.nanmin(error_map), np.nanmax(error_map))
    cmap = matplotlib.colormaps[cmap]
    colorized_error_map = cmap(((error_map - vmin) / (vmax - vmin)).clip(0, 1))[..., :3]
    if mask is not None:
        colorized_error_map = np.where(mask[..., None], colorized_error_map, 0)
    colorized_error_map = np.ascontiguousarray((colorized_error_map.clip(0, 1) * 255).astype(np.uint8))
    return colorized_error_map

def colorize_depth_affine(depth: np.ndarray, mask: np.ndarray = None, cmap: str = 'Spectral') -> np.ndarray:
    if mask is None:
        mask = np.logical_and((depth!=np.nan),(depth!=np.inf)).astype(np.bool_)

    #min_depth, max_depth = np.percentile(depth[mask], 1), np.percentile(depth[mask], 99)
    min_depth, max_depth = depth[mask].min(), depth[mask].max()
    depth = (depth - min_depth) / (max_depth - min_depth)
    depth = np.nan_to_num(depth,0)
    depth = np.clip(depth,0,1)
    colored = matplotlib.colormaps[cmap](depth)[..., :3]
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    mask = np.concatenate([mask[...,None],mask[...,None],mask[...,None]],axis=-1)
    colored = np.where(mask,colored,[0,0,0])
    return colored.astype(np.uint8)

def vis_af_mesh(depth: np.ndarray,
                       image: np.ndarray,
                       mask: np.ndarray,
                       K: list,
                       save_path: str = None,
                       show: bool= True):
    """
    depth: depth map 
    image: 8bit image
    K = [fx,fy,cx,cy]
    """
    H,W = depth.shape[:2]
    mask = np.ones_like(depth).astype(np.bool_)
    #unit K, only for utils3d toolkit
    gt_intrinsics = np.array([[K[0]/W,    0, 0.5], 
                            [   0, K[1]/H, 0.5], 
                            [   0,    0,    1]])
    points = utils3d.numpy.depth_to_points(depth, intrinsics=gt_intrinsics)
    normals, normals_mask = utils3d.numpy.points_to_normals(points, mask=mask)
    #Image.fromarray(colorize_normal(normals)).save("normal.png")

    faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
        points,
        image.astype(np.float32) / 255,
        utils3d.numpy.image_uv(width=W, height=H),
        mask=mask & ~(utils3d.numpy.depth_edge(depth, rtol=0.03, mask=mask) & utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
        tri=True
    )
    vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]

    mesh = trimesh.Trimesh(
                    vertices=vertices,
                    vertex_colors=vertex_colors,
                    faces=faces, 
                    process=False
    )
    if show:
        mesh.show()
    if save_path:
        mesh.export(save_path)


def vis_af_point_cloud(depth: np.ndarray,
                       image: np.ndarray,
                       mask: np.ndarray,
                       intrinsics: list,
                       save_path: str = None,
                       show: bool = True):
    """
    depth: depth map 
    image: 8bit image
    intrinsics:normalized intrinsics
    save_path: path to save point cloud
    """
    H,W = depth.shape[:2]
    mask = np.ones_like(depth).astype(np.bool_)
    #unit K, only for utils3d toolkit
    extrinsics = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=float)   # OpenGL's identity camera
    verts = utils3d.numpy.unproject_cv(utils3d.numpy.image_uv(*image.shape[:2]), depth, extrinsics=extrinsics, intrinsics=intrinsics)
    
    #depth_mask_ply = mask & (depth < depth[mask].min() * max_depth)
    point_cloud = trimesh.points.PointCloud(verts[mask], image[mask] / 255)
    if show:
        point_cloud.show()
    if save_path:
        point_cloud.export(save_path)


"""
import utils3d
from utils.data_utils import read_data
import numpy as np
import trimesh
import matplotlib
from PIL import Image

def colorize_normal(normal: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    if mask is not None:
        normal = np.where(mask[..., None], normal, 0)
    normal = normal * [0.5, -0.5, -0.5] + 0.5
    normal = (normal.clip(0, 1) * 255).astype(np.uint8)
    return normal

def colorize_error_map(error_map: np.ndarray, mask: np.ndarray = None, cmap: str = 'plasma', value_range: tuple[float, float] = None):
    vmin, vmax = value_range if value_range is not None else (np.nanmin(error_map), np.nanmax(error_map))
    cmap = matplotlib.colormaps[cmap]
    colorized_error_map = cmap(((error_map - vmin) / (vmax - vmin)).clip(0, 1))[..., :3]
    if mask is not None:
        colorized_error_map = np.where(mask[..., None], colorized_error_map, 0)
    colorized_error_map = np.ascontiguousarray((colorized_error_map.clip(0, 1) * 255).astype(np.uint8))
    return colorized_error_map

W = 1024
H = 768
FOCAL_LENGTH = 886.81
image_path = "test_sample/image/2.jpg"
gt_path = "test_sample/depth/2.hdf5"
gt_depth = read_data(gt_path)
min_depth = np.percentile(gt_depth,2)
max_depth = np.percentile(gt_depth,98)
#gt_depth = (gt_depth-min_depth)/(max_depth-min_depth)
#gt_depth=gt_depth*100
print(gt_depth.min(),gt_depth.max())
mask = np.ones_like(gt_depth).astype(np.bool_)
K = [886.81, 665.1, W/2, H/2]
gt_intrinsics = np.array([[K[0]/W,    0, 0.5], 
                          [   0, K[1]/H, 0.5], 
                          [   0,    0,    1]])
points = utils3d.numpy.depth_to_points(gt_depth, intrinsics=gt_intrinsics)
normals, normals_mask = utils3d.numpy.points_to_normals(points, mask=mask)
print(points.shape)
print(normals)
Image.fromarray(colorize_normal(normals)).save("1.png")

image = read_data(image_path)
print(image.shape)
faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
    points,
    image.astype(np.float32) / 255,
    utils3d.numpy.image_uv(width=W, height=H),
    mask=mask & ~(utils3d.numpy.depth_edge(gt_depth, rtol=0.03, mask=mask) & utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
    tri=True
)
vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]

mesh = trimesh.Trimesh(
                vertices=vertices,
                vertex_colors=vertex_colors,
                faces=faces, 
                process=False
)
mesh.show()
"""