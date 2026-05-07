import pytorch_lightning as pl
from model.utils import append_dims,count_params, instantiate_from_config
from model.diffusion_unet.latentcoder.autoencoder import VQModelInterface,DiagonalGaussianDistribution
from einops import rearrange
from torchvision.transforms import (
    InterpolationMode, 
    Resize, 
)
import numpy as np
import utils3d
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
from utils.metric import compute_metrics
from utils.geometry_utils import get_surface_normalv2
from typing import Tuple, Union
from os import PathLike
import torch
import torch.nn.functional as F
from enum import Enum
from utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift, angle_diff_vec3
class DiffusionBridgeSchedule:
    def __init__(self, T: Union[int, float] = 1.0,
                    sampling_steps: int = 10,
                    device: torch.device = "cpu"):
        self.T = T
        timesteps = torch.arange(T, -1, -T/ sampling_steps,device=device).round().int()
        self.timesteps = timesteps[:-1]
        self.next_timesteps = timesteps[1:]
        self.device = device

    def __len__(self) -> int:
        """
        Number of sampling steps.
        """
        return len(self.timesteps)

    def __getitem__(self, idx: Union[int, torch.IntTensor]) -> torch.Tensor:
        return self.timesteps[idx]

    def reset_timesteps(self,sampling_steps):
        timesteps = torch.arange(self.T, -1, -self.T/ sampling_steps,device=self.device).round().int()
        self.timesteps = timesteps[:-1]
        self.next_timesteps = timesteps[1:]

    def forward(self, x_0: torch.Tensor, x_T: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Diffusion bridge forward function.
        """
        #scale t to [0,1], in the belwo formula T = 1, so we ignore it.
        t = t / self.T
        t = t[(...,) + (None,) * (x_0.ndim - t.ndim)] if t.ndim < x_0.ndim else t
        return (1 - t) * x_0 + t * x_T + torch.sqrt(t*(1-t)) * torch.randn_like(x_0)
        
    @torch.no_grad()
    def backward(self, pred_x_0: torch.Tensor, x_t: torch.Tensor, x_T: torch.Tensor, t: torch.Tensor, next_t: torch.Tensor) -> torch.Tensor:
        """
        Diffusion bridge backward function.
        t --> t
        next_t --> t-1
        """
        if t==self.T or next_t==0:
            t = t / self.T
            next_t = next_t/self.T
            t = t[(...,) + (None,) * (pred_x_0.ndim - t.ndim)] if t.ndim < pred_x_0.ndim else t
            next_t = next_t[(...,) + (None,) * (pred_x_0.ndim - next_t.ndim)] if next_t.ndim < pred_x_0.ndim else next_t
            return (1 - next_t) * pred_x_0 + next_t * x_T
        else:
            t = t / self.T
            next_t = next_t/self.T
            t = t[(...,) + (None,) * (pred_x_0.ndim - t.ndim)] if t.ndim < pred_x_0.ndim else t
            next_t = next_t[(...,) + (None,) * (pred_x_0.ndim - next_t.ndim)] if next_t.ndim < pred_x_0.ndim else next_t
            ct1_noise = torch.sqrt(t*(1-t))
            pred_noise = (x_t - (1 - t) * pred_x_0 - t * x_T)/ct1_noise
            return (1 - next_t) * pred_x_0 + next_t * x_T + torch.sqrt(next_t*(1-next_t))*pred_noise

    @torch.no_grad()
    def euler_step(self, x_t: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor, next_t: torch.Tensor):
        t = t / self.T
        next_t = next_t/self.T
        dt = next_t - t
        dt = dt[(...,) + (None,) * (x_t.ndim - dt.ndim)] if dt.ndim < x_t.ndim else dt
        x_t = x_t + dt * v_pred
        return x_t
    
    def convert_to_pred(self, x_T: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
        return x_T - x_0
    
    def convert_from_pred(
        self, pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert from velocity prediction. Return predicted x_0 and x_T.
        """
        t = t / self.T
        t = t[(...,) + (None,) * (x_t.ndim - t.ndim)] if t.ndim < x_t.ndim else t
        B_t = t
        pred_x_0 = x_t - B_t * pred
        return pred_x_0



class ControlLDBM(pl.LightningModule):

    def __init__(
                self,
                model_config,
                first_stage_config,
                point_first_stage_config,
                control_stage_config,
                text_prompt_path,
                control_key = "depth",
                training_stage = "1",
                timesteps = 1000,
                sampling_steps = 10,
                ckpt_path=None,
                ignore_keys=[],
                point_vae_ckpt_path = None,
                control_ckpt_path = None,
                load_only_unet = False,
                learning_rate = 1e-4,
                scale_factor =0.,
                use_scheduler = False,
                scheduler_config = None,
                device = torch.device("cuda"),
                ):
        
        super().__init__()

        #self.loss_norm = loss_norm

        self.bridge_diffusion = DiffusionBridgeSchedule(T=timesteps, sampling_steps=sampling_steps, device=device)
        # use get the network
        self.timesteps = timesteps
        self.model = DiffusionWrapper(model_config)
        self.t_eps = 0.1
        count_params(self.model, verbose=True)


        #instantiate and load latent encoder and decoder
        self.scale_factor = scale_factor
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_point_first_stage(point_first_stage_config)
        self.instantiate_control_stage(control_stage_config)
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)
            #self.model.diffusion_model.copy_pretrained_weights_to_point_modules()
        if point_vae_ckpt_path is not None:
            self.init_point_stage_from_ckpt(point_vae_ckpt_path, ignore_keys=ignore_keys)
        if control_ckpt_path is not None:
            self.init_control_stage_from_ckpt(control_ckpt_path)

        #lr
        self.learning_rate    = learning_rate
        self.use_schedule     = use_scheduler
        self.scheduler_config = scheduler_config
        self.text_prompt_content = np.load(text_prompt_path)
        # 注册为 buffer 以便 Lightning 自动处理设备移动
        self.register_buffer("text_prompt", torch.from_numpy(self.text_prompt_content))

        self.control_key = control_key
        #self.control_scales = np.linspace(1.0, 0.1, 13).tolist()
        self.control_scales = [1.0] * 13
        self.training_stage = training_stage

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z
    
    def instantiate_point_first_stage(self, config):
        point_model = instantiate_from_config(config)
        self.point_first_stage_model = point_model.eval()
        for param in self.point_first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_control_stage(self, config):
        control_model = instantiate_from_config(config)
        self.control_model = control_model
        #for param in self.control_model.parameters():
            #param.requires_grad = False

    # make encoder and decoder
    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def decode_point_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.point_first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        if isinstance(self.point_first_stage_model, VQModelInterface):
            return self.point_first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
        else:
            return self.point_first_stage_model.decode(z)


    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        if isinstance(self.first_stage_model, VQModelInterface):
            return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
        else:
            return self.first_stage_model.decode(z)

    def decode_point_first_stage_for_train(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.point_first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        if isinstance(self.point_first_stage_model, VQModelInterface):
            return self.point_first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
        else:
            return self.point_first_stage_model.decode(z)


    def decode_first_stage_for_train(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        if isinstance(self.first_stage_model, VQModelInterface):
            return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
        else:
            return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_point_first_stage(self, x):
        return self.point_first_stage_model.encode(x)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def encode_first_stage_for_train(self, x):
        return self.first_stage_model.encode(x)

    # load model and return missing and unexpected
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        # load model state dict, if you want to load model
        # plz model.load_state_dict(torch.load(PATH))
        sd = torch.load(path, map_location="cpu", weights_only=False)
        # state_dict 回的是一个 OrderDict，存储了网络结构的名字和对应的参数
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik) or ik in k.split("."):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
    
    def init_point_stage_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik) or ik in k.split("."):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.point_first_stage_model.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")

    def init_control_stage_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik) or ik in k.split("."):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.control_model.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")


    #############
    #Train/Forward Part #
    #############
    @torch.no_grad()
    def get_input_k(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        #x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
        
    @torch.no_grad()
    def get_input(self, batch, return_pixel_outputs=False, is_normal=False): 
        """
        batch:
            point: [B,H,W,3]
            image: [B,H,W,3]
            caption: [B,1]
        return_first_stage_outputs:
            [original point map, reconstruction of point map]
        return_original_cond:
            [original image]
        """
        # encode point map before unet
        x_p = self.get_input_k(batch,"point")
        x_p = x_p.to(self.device)
        encoder_posterior_p = self.encode_point_first_stage(x_p)
        z_p = self.get_first_stage_encoding(encoder_posterior_p).detach()
        
        cond_text = torch.cat([self.text_prompt for _ in range(x_p.shape[0])], dim=0)
        #encode image
        x_img = self.get_input_k(batch, "image")
        x_img= x_img.to(self.device)
        encoder_posterior_img = self.encode_first_stage(x_img)
        z_img = self.get_first_stage_encoding(encoder_posterior_img).detach()

        control = self.get_input_k(batch, "control_geo")
        control = control.to(self.device)
        if self.control_key == "depth" or self.control_key == "canny":
            control = torch.cat([control, control, control],dim=1)
        encoder_posterior_c = self.encode_first_stage((control * 2.0 - 1.0))
        z_control = self.get_first_stage_encoding(encoder_posterior_c).detach()

        if is_normal:
            x_normal = self.get_input_k(batch, "normal")
            x_normal= x_normal.to(self.device)
            encoder_posterior_normal = self.encode_first_stage(x_normal)
            z_normal = self.get_first_stage_encoding(encoder_posterior_normal).detach()
        else:
            x_normal = x_img
            z_normal = z_img

        mask = self.get_input_k(batch,"depth_mask").to(self.device)
        latent_out = [z_p, z_normal, z_img, z_control, cond_text]
        pixel_out  = [x_img, control]

        if return_pixel_outputs:
            rec_point = self.decode_point_first_stage(z_p)
            rec_normal = self.decode_first_stage(z_normal)
            rec_image = self.decode_first_stage(z_img)
            #x_p_rec = (x_p_rec+1.0)/2.0
            pixel_out.extend([x_p, x_normal, rec_image, rec_point, rec_normal])
            return latent_out,pixel_out,mask
        return latent_out,mask

    def get_loss(self, pred, target):
        loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        return loss

    def training_bridge_losses(self, z_p_0, control, z_T, t, cond_img, cond_text, x_p, x_normal):
        terms = {}
        z_0 = torch.cat([z_p_0],dim=1)
        z_t   = self.bridge_diffusion.forward(z_0, z_T, t)
        pred_z0  = self.denoise(z_t, control, t.long(), cond_img, cond_text)
        pred_v  = (z_T - pred_z0)
        gt_v = self.bridge_diffusion.convert_to_pred(z_T, z_0)
        mse = self.get_loss(pred_v, gt_v)
        terms["train/loss"] = mse.mean() 
        return terms["train/loss"] , terms
        
    def forward(self,batch):
        latent_out, pixel_out, mask = self.get_input(batch, return_pixel_outputs=True)
        x_img, control, x_p, x_normal, rec_image, rec_point, rec_normal = pixel_out
        z_p, z_normal, z_img, z_control, cond_text = latent_out
        z_p_0  = z_p
        z_T    = torch.cat([z_img],dim=1)
        t = torch.randint(0, self.timesteps+1, (z_p_0.shape[0],), device=z_T.device)
        return self.training_bridge_losses(z_p_0, z_control, z_T, t, z_img, cond_text, x_p, x_normal)

    def denoise(
        self,
        z_t,
        control,
        t,
        cond_img=None,
        cond_text=None
    ):
        if not isinstance(cond_img,list):
            cond_img = [cond_img]
        if not isinstance(cond_text,list):
            cond_text = [cond_text]

        z_t = torch.cat([z_t] + cond_img, dim=1)
        control_feat = self.control_model(x=cond_img[0], 
                                          hint = control, 
                                          timesteps = t.to(z_t.device), 
                                          context=cond_text)
        control_feat = [(c * scale).to(z_t.device) for c, scale in zip(control_feat, self.control_scales)]
        model_output = self.model(z_t, t, control_feat, cond_text)
        return model_output
    


    @torch.no_grad()
    def backward_sample(self, cond_img, control, cond_text):
        z_t = torch.cat([cond_img],dim=1)
        z_T = torch.cat([cond_img],dim=1)
        for i in range(len(self.bridge_diffusion.timesteps)):
            t = self.bridge_diffusion.timesteps[i]
            next_t  = self.bridge_diffusion.next_timesteps[i]
            pred_x0 = self.denoise(z_t, control, t, cond_img, cond_text)
            z_t   = self.bridge_diffusion.backward(pred_x0, x_t = z_t, x_T = z_T, t=t, next_t = next_t)
        return z_t

    @torch.no_grad()
    def backward_sample2(self, cond_img, control, cond_text):
        z_t = torch.cat([cond_img],dim=1)
        z_T = torch.cat([cond_img],dim=1)
        for i in range(len(self.bridge_diffusion.timesteps)):
            t = self.bridge_diffusion.timesteps[i]
            next_t  = self.bridge_diffusion.next_timesteps[i]
            pred_x0 = self.denoise(z_t, control, t, cond_img, cond_text)
            v_pred = z_T - pred_x0
            z_t   = self.bridge_diffusion.euler_step(x_t=z_t, v_pred=v_pred, t=t, next_t=next_t)
        return z_t



    def point_to_geometry(self,point_map: torch.Tensor,fov_x=None,force_projection=False):
        """
        point_map --> torch.tensor. [B,H,W,C]
        Output:
            normal: torch.tensor. [B,3,H,W]
        """
        original_height, original_width = point_map.shape[-2:]
        aspect_ratio = original_width / original_height
        if point_map.shape[1]==3:
            point_map = rearrange(point_map,'b c h w -> b h w c')
        mask = torch.ones_like(point_map).bool()
        points, mask, fov_x = map(lambda x: x.float() if isinstance(x, torch.Tensor) else x, [point_map, mask, fov_x])
        normal,mask_normal = get_surface_normalv2(points)
        
        return normal

    @torch.no_grad()
    def log_images(
        self, 
        batch, 
        N=8, 
        **kwargs
    ):

        log = dict()
        latent_out, pixel_out, mask= self.get_input(batch, 
                            return_pixel_outputs=True)
        z_p, z_normal, z_img, z_control, cond_text = latent_out
        x_img, control, x_p, x_normal, rec_image, rec_point, rec_normal = pixel_out
        gt_depth = batch['depth'].squeeze(1)
        #get gt
        N = min(z_p.shape[0], N)
        #[B,C,H,W] -> [B,H,W,C]
        log["rgb_image"] = x_img[:N].permute(0, 2, 3, 1)
        log["gt_point_map"] = x_p[:N].permute(0, 2, 3, 1)
        log["gt_depth"]  = gt_depth[:N] #depth  [B,H,W]
        x_normal = self.point_to_geometry(x_p)
        log["gt_normal"] = x_normal[:N].permute(0, 2, 3, 1) 

        rec_point = rec_point[:N].clamp(-1,1)
        rec_image = rec_image[:N]
        
        # get prediction
        pred_latent_map = self.backward_sample(z_img[:N], z_control[:N], cond_text[:N])
        pred_latent_point_map = pred_latent_map[:,:4,...]
        pred_point_map = self.decode_point_first_stage(pred_latent_point_map)
        pred_point_map = pred_point_map.clamp(-1,1)
        
        WIDTH, HEIGHT = batch["width"][0], batch["height"][0]
        H,W = pred_point_map.shape[-2],pred_point_map.shape[-1]
        if H != HEIGHT or W !=WIDTH:
            resize_transform = Resize(
                            size=(HEIGHT, WIDTH), interpolation=InterpolationMode.BILINEAR
                        )
            pred_point_map = resize_transform(pred_point_map)
            rec_point = resize_transform(rec_point)
            rec_image = resize_transform(rec_image)

        rec_depth = rec_point[:, 2, :, :]
        
        pred_normal_map = self.point_to_geometry(pred_point_map)
        pred_depth = pred_point_map[:, 2, :, :]

        log["pred_point_map"] = pred_point_map.permute(0, 2, 3, 1)
        log["pred_depth"] = pred_depth
        log["pred_normal"] = pred_normal_map.permute(0, 2, 3, 1) 

        rec_normal = self.point_to_geometry(rec_point)
        log["rec_point"]  = rec_point.permute(0, 2, 3, 1) 
        log["rec_normal"] = rec_normal.permute(0, 2, 3, 1) 
         
        log["rec_depth"]  = rec_depth
        log["rec_image"]  = rec_image.permute(0, 2, 3, 1) 

        if self.control_key == "depth" or self.control_key == "canny":
            log["control_map"] = control[:N,0,...]
        else:
            log["control_map"] = control[:N]

        return log

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_schedule:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss
    

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pred = {}
        latent_out ,mask = self.get_input(batch, is_normal = False)
        z_p, z_normal, z_img, z_control, cond_text = latent_out
        
        pred_latent_map = self.backward_sample2(z_img, z_control, cond_text)
        pred_latent_point_map, pred_latent_normal_map = pred_latent_map[:,:4,...], pred_latent_map[:,4:,...]
        pred_point_map = self.decode_point_first_stage(pred_latent_point_map)
        #pred_point_map = (pred_point_map+1.0)/2.0
        pred_point_map = pred_point_map.clamp(-1,1)
        WIDTH, HEIGHT = batch["width"][0], batch["height"][0]
        H,W = pred_point_map.shape[-2],pred_point_map.shape[-1]
        if H != HEIGHT or W !=WIDTH:
            resize_transform = Resize(
                            size=(HEIGHT, WIDTH), interpolation=InterpolationMode.BILINEAR
                        )
            pred_point_map = resize_transform(pred_point_map)
        pred_depth = pred_point_map[:,2,...]
        
        pred['point'] = pred_point_map
        pred['depth'] = pred_depth
        metrics,_,_ = compute_metrics(pred, batch)

        self.log("val/point_abs_rel", metrics['points_affine_invariant']['rel'], 
                on_step=True, on_epoch=True, prog_bar=True, 
                batch_size=1, sync_dist=True)
        self.log("val/point_d1", metrics['points_affine_invariant']['delta1'], 
                on_step=True, on_epoch=True, prog_bar=True, 
                batch_size=1, sync_dist=True)
        self.log("val/depth_abs_rel", metrics['depth_affine_invariant']['rel'], 
                on_step=True, on_epoch=True, prog_bar=True, 
                batch_size=1, sync_dist=True)
        self.log("val/depth_d1", metrics['depth_affine_invariant']['delta1'], 
                on_step=True, on_epoch=True, prog_bar=True, 
                batch_size=1, sync_dist=True)
        
        return {
            "point_abs_rel": metrics['points_affine_invariant']['rel'],
            "point_d1": metrics['points_affine_invariant']['delta1'],
            "depth_abs_rel": metrics['depth_affine_invariant']['rel'],
            "depth_d1": metrics['depth_affine_invariant']['delta1']
        }

    def get_infer_device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def normalize_infer_size(self, size):
        if isinstance(size, int):
            return (size, size)
        if len(size) != 2:
            raise ValueError("size must be an int or a two-item tuple/list in (height, width) order.")
        return (int(size[0]), int(size[1]))

    def read_infer_image_source(self, source, convert_rgb=False):
        if isinstance(source, (str, bytes, PathLike)):
            with Image.open(source) as image:
                image = image.convert("RGB") if convert_rgb else image
                return np.asarray(image).copy()
        if isinstance(source, Image.Image):
            image = source.convert("RGB") if convert_rgb else source
            return np.asarray(image).copy()
        return source

    def convert_infer_source_to_bchw(self, source, device, convert_rgb=False):
        source = self.read_infer_image_source(source, convert_rgb=convert_rgb)

        if isinstance(source, torch.Tensor):
            tensor = source.detach().to(device=device, dtype=torch.float32)
        elif isinstance(source, np.ndarray):
            tensor = torch.from_numpy(source).to(device=device, dtype=torch.float32)
        else:
            raise TypeError(
                "infer input must be a path, PIL image, numpy array, or torch tensor; "
                f"got {type(source)}."
            )

        if tensor.ndim == 2:
            tensor = tensor[None, None, ...]
        elif tensor.ndim == 3:
            if tensor.shape[0] in (1, 3, 4):
                tensor = tensor[None, ...]
            else:
                tensor = tensor.permute(2, 0, 1)[None, ...]
        elif tensor.ndim == 4:
            if tensor.shape[1] not in (1, 3, 4):
                tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"infer input must have 2, 3, or 4 dimensions; got {tensor.ndim}.")

        return tensor.contiguous()

    def normalize_infer_value_range(self, tensor):
        if tensor.numel() == 0:
            return tensor
        max_value = tensor.detach().amax()
        if max_value > 1.0:
            scale = 65535.0 if max_value > 255.0 else 255.0
            tensor = tensor / scale
        return tensor.clamp(0.0, 1.0)

    def match_infer_channels(self, tensor, channels):
        current_channels = tensor.shape[1]
        if current_channels == channels:
            return tensor
        if current_channels == 1 and channels == 3:
            return tensor.repeat(1, 3, 1, 1)
        if current_channels > channels:
            return tensor[:, :channels, ...]
        raise ValueError(f"Cannot convert infer tensor from {current_channels} to {channels} channels.")

    def resize_infer_tensor(self, tensor, size, mode="bilinear"):
        if tensor.shape[-2:] == size:
            return tensor
        align_corners = False if mode in ("linear", "bilinear", "bicubic", "trilinear") else None
        return F.interpolate(tensor, size=size, mode=mode, align_corners=align_corners)

    def prepare_infer_image(self, image, size):
        device = self.get_infer_device()
        target_size = self.normalize_infer_size(size)
        image_tensor = self.convert_infer_source_to_bchw(image, device=device, convert_rgb=True)
        original_size = image_tensor.shape[-2:]
        image_tensor = self.match_infer_channels(image_tensor, 3)
        image_tensor = self.normalize_infer_value_range(image_tensor)
        image_tensor = self.resize_infer_tensor(image_tensor, target_size)
        return image_tensor * 2.0 - 1.0, original_size

    def prepare_infer_control(self, control, size, batch_size):
        device = self.get_infer_device()
        target_size = self.normalize_infer_size(size)
        control_tensor = self.convert_infer_source_to_bchw(control, device=device, convert_rgb=False)
        control_tensor = self.normalize_infer_value_range(control_tensor)
        control_tensor = self.resize_infer_tensor(control_tensor, target_size)

        if self.control_key in ("depth", "canny") or control_tensor.shape[1] == 1:
            control_tensor = self.match_infer_channels(control_tensor, 3)

        if control_tensor.shape[0] == 1 and batch_size > 1:
            control_tensor = control_tensor.expand(batch_size, -1, -1, -1)
        if control_tensor.shape[0] != batch_size:
            raise ValueError(
                f"control batch size {control_tensor.shape[0]} does not match image batch size {batch_size}."
            )

        return control_tensor * 2.0 - 1.0

    def encode_infer_condition(self, image, control):
        image_z = self.get_first_stage_encoding(self.encode_first_stage(image)).detach()
        control_z = self.get_first_stage_encoding(self.encode_first_stage(control)).detach()
        cond_text = torch.cat([self.text_prompt for _ in range(image_z.shape[0])], dim=0)
        cond_text = cond_text.to(device=image_z.device, dtype=image_z.dtype)
        return image_z, control_z, cond_text

    def prepare_infer_mask(self, mask, target_size, device):
        if mask is None:
            return None
        mask_tensor = self.convert_infer_source_to_bchw(mask, device=device, convert_rgb=False)
        if mask_tensor.shape[1] > 1:
            mask_tensor = mask_tensor[:, :1, ...]
        mask_tensor = self.resize_infer_tensor(mask_tensor.float(), target_size, mode="nearest")
        return mask_tensor[:, 0, ...] > 0

    def recover_infer_camera_points(self, point_map, mask=None):
        height, width = point_map.shape[-2:]
        aspect_ratio = width / height
        points = point_map.permute(0, 2, 3, 1)
        valid_mask = self.prepare_infer_mask(mask, (height, width), point_map.device)

        focal, shift, offset = recover_focal_shift(points, valid_mask)
        points_corrected = points.clone()
        points_corrected[..., :2] = points[..., :2] + offset[..., None, None, :]
        depth = points_corrected[..., 2] + shift[..., None, None]

        diagonal_scale = (1 + aspect_ratio ** 2) ** 0.5
        fx = focal / 2 * diagonal_scale / aspect_ratio
        fy = focal / 2 * diagonal_scale
        center_x = torch.tensor(0.5, device=points.device, dtype=points.dtype)
        center_y = torch.tensor(0.5, device=points.device, dtype=points.dtype)
        intrinsics = utils3d.pt.intrinsics_from_focal_center(fx, fy, center_x, center_y)
        return utils3d.pt.depth_map_to_point_map(depth, intrinsics=intrinsics).permute(0, 3, 1, 2)

    @torch.no_grad()
    def infer(self, image_path, control, infer_steps=10, size=(640, 480), return_normal=False, mask=None):
        """
        Run single-image or batched inference.

        Args:
            image_path: Path, PIL image, numpy array, or tensor. Image tensors can be HWC, CHW, BHWC, or BCHW.
            control: Path, PIL image, numpy array, or tensor. Values can be in [0, 1], [0, 255], or [0, 65535].
            infer_steps: Number of bridge sampling steps.
            size: Inference resize in (height, width) order.
            return_normal: Kept for API compatibility. The normal map is always returned.
            mask: Optional valid-pixel mask used by camera recovery.
        """
        image, original_size = self.prepare_infer_image(image_path, size)
        control = self.prepare_infer_control(control, size, batch_size=image.shape[0])
        image_z, control_z, cond_text = self.encode_infer_condition(image, control)

        self.bridge_diffusion.reset_timesteps(infer_steps)
        pred_latent_map = self.backward_sample2(image_z, control_z, cond_text)
        pred_latent_point_map = pred_latent_map
        pred_point_map = self.decode_point_first_stage(pred_latent_point_map).clamp(-1, 1)
        pred_point_map = self.recover_infer_camera_points(pred_point_map, mask=mask)
        pred_point_map = self.resize_infer_tensor(pred_point_map, original_size)
        pred_point_map = rearrange(pred_point_map, "b c h w -> b h w c")

        pred_depth = pred_point_map[..., 2]
        pred_normal_map = self.point_to_geometry(pred_point_map)

        return pred_point_map.detach().cpu(), pred_depth.detach().cpu(), pred_normal_map.detach().cpu()

    @torch.no_grad()
    def evaluate_infer(self,batch):
        """
        image: image path
        infer_steps: int -->backward steps for implicit bridge sampling,default is 410
        rho: float -->implicit bridge sampling parameter,default is 0.0 for deterministic sampling
        size: tuple -->image size for resize,default is (512,512)
        """
        latent_out , mask = self.get_input(batch, is_normal=False)
        z_p, z_normal, z_img, z_control, cond_text = latent_out
        pred = {}
        
        #self.bridge_diffusion.reset_timesteps(infer_steps)
        pred_latent_map = self.backward_sample2(z_img, z_control,cond_text)
        pred_latent_point_map = pred_latent_map[:,:4,...]
        pred_point_map = self.decode_point_first_stage(pred_latent_point_map)
        pred_point_map = pred_point_map.clamp(-1,1)

        original_height, original_width = pred_point_map.shape[-2:]
        aspect_ratio = original_width / original_height

        pred_point_map = pred_point_map.permute(0,2,3,1)


        focal, shift, offset = recover_focal_shift(pred_point_map, mask)
        points_corrected = pred_point_map.clone()
        points_corrected[..., :2] = pred_point_map[..., :2] + offset[..., None, None, :]
        #pred_point_map[..., 2] += shift[..., None, None]
        depth = points_corrected[..., 2] + shift[..., None, None]
        #print(focal,shift)
        fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 
        intrinsics = utils3d.pt.intrinsics_from_focal_center(fx, fy, torch.tensor(0.5, device=pred_point_map.device, dtype=pred_point_map.dtype), torch.tensor(0.5, device=pred_point_map.device, dtype=pred_point_map.dtype))
        #sprint(intrinsics)
        pred_point_map = utils3d.pt.depth_map_to_point_map(depth, intrinsics=intrinsics)
        pred_point_map = pred_point_map.permute(0,3,1,2)

        
        WIDTH, HEIGHT = batch["width"], batch["height"]
        H,W = pred_point_map.shape[-2],pred_point_map.shape[-1]
        if H != HEIGHT or W !=WIDTH:
            resize_transform = Resize(
                            size=(HEIGHT, WIDTH), interpolation=InterpolationMode.BILINEAR
                        )
            pred_point_map = resize_transform(pred_point_map)

        pred_depth = pred_point_map[:, 2, :, :]

        pred['point'] = pred_point_map
        pred['depth'] = pred_depth
        metrics = compute_metrics(pred, batch)

        return pred_point_map.permute(0, 2, 3, 1).detach().cpu(), pred_depth.detach().cpu(), metrics

    
    def configure_optimizers(self):
        lr = self.learning_rate
        #params = list(self.model.parameters())
        params = list(self.model.parameters())+list(self.control_model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_schedule:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)

    def forward(self, x, t, control: list = None, c_crossattn: list = None):
        cc = torch.cat(c_crossattn, 1)
        out= self.diffusion_model(x, t, context=cc, control=control)
        return out