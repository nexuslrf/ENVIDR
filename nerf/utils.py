import os
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
import lpips

import time

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]])

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]])

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]])

def get_one_pose(device, radius=2, theta=1*np.pi/2, phi=np.pi):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''

    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.tensor([theta], device=device)
    phis = torch.tensor([phi], device=device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    size = 1
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None, patch_size=1, center_crop=0, center_crop_ratio=0.6):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        # if use patch-based sampling, ignore error_map
        if patch_size > 1:

            # random sample left-top cores.
            # NOTE: this impl will lead to less sampling on the image corner pixels... but I don't have other ideas.
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten

            inds = inds.expand([B, N])

        elif error_map is None:
            if center_crop <= 0:
                inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
                # inds = torch.randperm(H*W, device=device)[:N]
                inds = inds.expand([B, N])
            else:
                pad = (1 - center_crop) / 2
                H_pad, W_pad = int(H * pad), int(W * pad)
                H_crop, W_crop = H - H_pad * 2, W - W_pad * 2
                N_crop = int(N * center_crop_ratio)
                i_crop = torch.randint(0, H_crop, size=[N_crop], device=device) + H_pad
                j_crop = torch.randint(0, W_crop, size=[N_crop], device=device) + W_pad
                inds_crop = i_crop * W + j_crop
                inds_bg = torch.randint(0, H*W, size=[N - N_crop], device=device)
                inds = torch.cat([inds_crop, inds_bg], dim=0)
                # unique
                inds = torch.unique(inds)
                inds = inds.expand([B, -1])
                # import IPython; IPython.embed()
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
        
    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N, 3] or [B, H, W, 3], range[0, 1]
          
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'

class LPIPSMeter:
    def __init__(self, net='alex', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item() # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1
    
    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'

class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.loaded_from_ckpt = False
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.load_renv = False

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        # optionally use LPIPS loss for patch-based training
        if self.opt.patch_size > 1:
            import lpips
            self.criterion_lpips = lpips.LPIPS(net='alex').to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "env_checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)   

            f = os.path.join(self.workspace, 'args.ini')
            with open(f, 'w') as file:
                for arg in sorted(vars(opt)):
                    attr = getattr(opt, arg)
                    file.write('{} = {}\n'.format(arg, attr))
            if opt.config is not None:
                f = os.path.join(self.workspace, 'config.ini')
                os.system(f'cp {opt.config} {f}')
            if opt.env_dataset_config is not None and opt.env_dataset_config != "":
                f = os.path.join(self.workspace, os.path.basename(opt.env_dataset_config))
                os.system(f'cp {opt.env_dataset_config} {f}')                
            if self.model.env_opt is not None and self.model.env_opt.env_images_list:
                f = os.path.join(self.workspace, os.path.basename(self.model.env_opt.env_images_list))
                os.system(f'cp {self.model.env_opt.env_images_list} {f}')   

            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.env_ckpt_path = os.path.join(self.ckpt_path, 'env_ckpts')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            if self.opt.env_sph_mode:
                self.best_env_paths = []
                for env_name in self.model.env_opt.env_images_names:
                    self.best_env_paths.append(f"{self.env_ckpt_path}/{self.name}_{env_name}.pth")
            else:
                self.best_env_path = f"{self.env_ckpt_path}/{self.name}_env_net.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
            os.makedirs(self.env_ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] options: {self.opt}')
        if self.model.env_opt is not None:
            self.log(f'[INFO] env options: {self.model.env_opt}')
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint, model_only=opt.ckpt_model_only)
        
        if opt.color_mlp_path != "" and not opt.test:
            print(f"loading color mlp from: {opt.color_mlp_path}")
            if not opt.train_renv and not opt.train_env_only:
                state_dict = torch.load(opt.color_mlp_path, map_location=self.device)
                color_state = state_dict['model']            
                if 'specular' in opt.resume_mlps:
                    net_prefix = 'color_net.'
                    net_state = {k[len(net_prefix):]: v for k, v in color_state.items() if k.startswith(net_prefix)}
                    self.model.color_net.load_state_dict(net_state)
                if 'diffuse' in opt.resume_mlps:
                    net_prefix = 'diffuse_net.'
                    net_state = {k[len(net_prefix):]: v for k, v in color_state.items() if k.startswith(net_prefix)}
                    self.model.diffuse_net.load_state_dict(net_state)
                if 'renv' in opt.resume_mlps and opt.use_renv and not self.load_renv:
                    net_prefix = 'renv_net.'
                    net_state = {k[len(net_prefix):]: v for k, v in color_state.items() if k.startswith(net_prefix)}
                    try:
                        self.model.renv_net.load_state_dict(net_state)
                    except:
                        print("renv_net not found in ckpt, skip loading")
            else:
                self.load_checkpoint(self.opt.color_mlp_path, model_only=True)

        # clip loss prepare
        if opt.rand_pose >= 0: # =0 means only using CLIP loss, >0 means a hybrid mode.
            from nerf.clip_utils import CLIPLoss
            self.clip_loss = CLIPLoss(self.device)
            self.clip_loss.prepare_text([self.opt.clip_text]) # only support one text prompt now...

        if len(opt.marching_aabb) == 6:
            device = self.model.aabb_train.device
            self.model.aabb_train = torch.FloatTensor(opt.marching_aabb).to(device) * opt.scale
            self.model.aabb_train = self.model.aabb_train.clamp(min=-opt.bound, max=opt.bound)
            self.model.aabb_infer = self.model.aabb_train.clone()
            self.log(f'[INFO] aabb_marching: {self.model.aabb_train}')

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	

    def train_step(self, data, loader=None):
        r_images = None
        if 'poses' in data:
            poses = data['poses'].to(self.device)
            rays = get_rays(poses, loader._data.intrinsics, loader._data.H, loader._data.W, loader._data.num_rays, loader._data.error_map, loader._data.opt.patch_size)
            rays_o = rays['rays_o']
            rays_d = rays['rays_d']
            images = data['images'].to(self.device)
            C = images.shape[-1]
            images = torch.gather(images.view(1, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            data['images'] = images
            if 'diffuse_images' in data:
                diffuse_images = data['diffuse_images'].to(self.device)
                diffuse_images = torch.gather(diffuse_images.view(1, -1, C), 1, \
                                              torch.stack(diffuse_images.shape[-1] * [rays['inds']], -1)) # [B, N, 3/4]
                data['diffuse_images'] = diffuse_images
                if self.opt.diffuse_only:
                    data['images'] = diffuse_images
            if 'r_images' in data:
                r_images = data['r_images'].to(self.device)
                C = r_images.shape[-1]
                r_images = torch.gather(r_images.view(1, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
                data['r_images'] = r_images
            else:
                r_images = None
        else:
            rays_o = data['rays_o'].to(self.device) # [B, N, 3]
            rays_d = data['rays_d'].to(self.device) # [B, N, 3]

        env_net_index = None
        if "env_net_indices" in data:
            env_net_indices = data["env_net_indices"]
            # TODO: assume batch size is 1
            assert env_net_indices.shape[0] == 1
            env_net_index = env_net_indices[0]

        # if there is no gt image, we train with CLIP loss.
        if 'images' not in data:

            B, N = rays_o.shape[:2]
            H, W = data['H'], data['W']

            # currently fix white bg, MUST force all rays!
            outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=None, perturb=True, force_all_rays=True, get_normal_image=False, **vars(self.opt))
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()

            # [debug] uncomment to plot the images used in train_step
            #torch_vis_2d(pred_rgb[0])

            loss = self.clip_loss(pred_rgb)
            
            return pred_rgb, None, loss

        images = data['images'] #.to(self.device) # [B, N, 3/4]
        material = data['material'] if 'material' in data else None
        diffuse_images = data['diffuse_images'] if 'diffuse_images' in data else None

        B, N, C = images.shape

        if B > 1: # for the ease of coding, we assume B=1.
            # import IPython; IPython.embed()
            N = N*B
            rays_o = rays_o.reshape(1, N, 3)
            rays_d = rays_d.reshape(1, N, 3)
            images = images.reshape(1, N, C)
            
        alpha_mask = None
        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])
            if diffuse_images:
                diffuse_images[..., :3] = srgb_to_linear(diffuse_images[..., :3])

        if C == 3 or self.model.bg_radius > 0:
            bg_color = 1
        # train with random background color if not using a bg model and has alpha channel.
        else:
            if self.opt.alpha_bg_mode == 'white':
                bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            #bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            if self.opt.alpha_bg_mode == 'random':
                bg_color = torch.rand_like(images[..., :3]) # [N, 3], pixel-wise random.
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
            alpha_mask = images[..., 3] # 0 is transparent, 1 is opaque.
        else:
            gt_rgb = images
            
        if self.opt.use_neus_sdf:
            cos_anneal_steps = self.opt.cos_anneal_steps
            self.opt.cos_anneal_ratio = 1.0 if cos_anneal_steps == 0 else min(1.0, self.global_step / cos_anneal_steps)

        use_specular_color = self.opt.color_net_start_iter == 0 or self.epoch > self.opt.color_net_start_iter 
        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, get_normal_image=False, perturb=True, \
                force_all_rays=False if self.opt.patch_size == 1 else True, use_specular_color=use_specular_color, \
                env_net_index=env_net_index, material=material, r_images=r_images, **vars(self.opt))
        # outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=True, **vars(self.opt))
    
        pred_rgb = outputs['image']

        # MSE loss (or L1 loss)
        color_loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]
        loss = self.opt.color_loss_weight * color_loss
        # patch-based rendering
        if False and self.opt.patch_size > 1:
            gt_rgb = gt_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()
            pred_rgb = pred_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()
            # import IPython; IPython.embed()
            # torch_vis_2d(gt_rgb[0])
            # torch_vis_2d(pred_rgb[0])

            # LPIPS loss [not useful...]
            loss = loss + 1e-3 * self.criterion_lpips(pred_rgb, gt_rgb)

        # special case for CCNeRF's rank-residual training
        if len(loss.shape) == 3: # [K, B, N]
            loss = loss.mean(0)

        # update error_map
        if self.error_map is not None and 'inds_coarse' in data:
            index = data['index'] # [B]
            inds = data['inds_coarse'] # [B, N]

            # take out, this is an advanced indexing and the copy is unavoidable.
            error_map = self.error_map[index] # [B, H * W]

            # [debug] uncomment to save and visualize error map
            # if self.global_step % 1001 == 0:
            #     tmp = error_map[0].view(128, 128).cpu().numpy()
            #     print(f'[write error map] {tmp.shape} {tmp.min()} ~ {tmp.max()}')
            #     tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            #     cv2.imwrite(os.path.join(self.workspace, f'{self.global_step}.jpg'), (tmp * 255).astype(np.uint8))

            error = loss.detach().to(error_map.device) # [B, N], already in [0, 1]
            
            # ema update
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

            # put back
            self.error_map[index] = error_map

        loss = loss.mean()
        loss_dict = dict()
        loss_dict['color'] = loss.detach()

        if 'empty' in outputs:
            return pred_rgb, gt_rgb, loss, loss_dict

        if self.opt.diffuse_loss and self.opt.diffuse_loss_weight > 0 and diffuse_images is not None:
            diffuse_loss = self.criterion(outputs['diffuse_image'], diffuse_images).mean()
            loss = loss + self.opt.diffuse_loss_weight * diffuse_loss
            loss_dict['diffuse'] = diffuse_loss.detach()

        if self.opt.mask_loss and self.opt.mask_loss_weight > 0 and alpha_mask is not None:
            weights_sum = outputs['weights_sum'] # 0 is bg, 1 is fg
            # BCE
            loss_mask = F.binary_cross_entropy(weights_sum.clip(1e-3, 1.0 - 1e-3), alpha_mask)
            loss = loss + self.opt.mask_loss_weight * loss_mask
            loss_dict['mask'] = loss_mask.detach()


        if self.opt.relsdf_loss and self.opt.relsdf_loss_weight > 0:
            relsdf = outputs['relsdf']
            est_relsdf = outputs['est_relsdf']
            l_d = torch.square(relsdf - est_relsdf)
            l_d = l_d.mean() if self.opt.relsdf_mode == 'mean' else l_d.sum() # 
            loss = loss + self.opt.relsdf_loss_weight * l_d
            loss_dict['relsdf'] = l_d.detach()
        
        if self.opt.dist_bound and self.opt.dist_bound_weight > 0:
            relsdf = outputs['relsdf']
            dist = outputs['sdf_dist']
            l_db = (torch.relu(relsdf.abs() - dist)**2).sum()
            loss = loss + self.opt.dist_bound_weight * l_db
            loss_dict['distbnd'] = l_db.detach()

        # TODO: back-sdf
        if self.opt.backsdf_loss and self.opt.backsdf_loss_weight > 0:
            weights = outputs['sdf_weights']
            weight_mask = weights > self.opt.backsdf_thresh
            sdf_mask = outputs['relsdf'] > 0
            mask = weight_mask * sdf_mask
            s_sq = outputs['relsdf'][mask]**2
            masked_weight = weights[mask]
            r_cos_sq = s_sq / (outputs['sdf_dist'][mask].clamp(min=5e-4)**2 + s_sq)
            denom = 1 if self.opt.backsdf_mode == 'sum' else (1+masked_weight.sum())
            l_b = (masked_weight * r_cos_sq).sum() / denom
            loss = loss + self.opt.backsdf_loss_weight * l_b
            loss_dict['backsdf'] = l_b.detach()

        if self.opt.orientation_loss and self.opt.orientation_loss_weight > 0:
            weights = outputs['sdf_weights']
            cos = outputs['cos']
            # weight_mask = weights > 0.005
            l_o = (weights * torch.relu(cos)).sum()
            loss = loss + self.opt.orientation_loss_weight * l_o
            loss_dict['orientation'] = l_o.detach()

        reg_density = None
        if self.opt.cauchy_loss and self.opt.cauchy_loss_weight > 0:
            sdfs = outputs['sdfs']
            beta = self.model.sdf_density.get_beta()
            beta = beta.detach() if not self.opt.cauchy_undetach_beta else beta
            reg_density = self.model.sdf_density.density_func(sdfs, beta, 1)
            loss_scale = 4.0
            roughness_weight = 1
            if self.opt.cauchy_roughness_weighted and 'roughness' in outputs:
                roughness = outputs['roughness'].detach().squeeze()
                roughness_weight = torch.sigmoid((0.5*((1 / roughness.clamp(min=2e-2, max=0.1)) - 25))) * 10
                # import IPython; IPython.embed()

            cauchy_loss = 1.0 / loss_scale * (torch.log1p((1-reg_density)**2 * (loss_scale **2)) * roughness_weight).mean()
            loss = loss + self.opt.cauchy_loss_weight * cauchy_loss
            loss_dict['cauchy'] = cauchy_loss.detach()
        # print("beta={}, loss={}".format(self.model.sdf_density.beta, loss)) # (1)

        if self.opt.entropy_loss_weight > 0:
            if reg_density is None:
                sdfs = outputs['sdfs']
                beta = self.model.sdf_density.get_beta()
                beta = beta.detach() if not self.opt.cauchy_undetach_beta else beta
                reg_density = self.model.sdf_density.density_func(sdfs, beta, 1)
            weights = outputs['sdf_weights']
            weight_mask = weights > 0.02
            density_mask = (reg_density > 0.05) * (reg_density < 0.95)
            mask = weight_mask * density_mask
            entropy_loss = ((-torch.log(1 - reg_density + 1e-6)) * mask * weights).sum()
            loss = loss + self.opt.entropy_loss_weight * entropy_loss
            loss_dict['entropy'] = entropy_loss.detach()

        if self.opt.eikonal_loss and self.opt.eikonal_loss_weight > 0:
            sdf_gradients = outputs['sdf_gradients']
            # print("sdf_gradients.shape={}".format(sdf_gradients.shape)) # (1, 3, 128, 128)
            eikonal_loss = ((sdf_gradients.norm(p=2, dim=-1) - 1) ** 2).mean()
            loss = loss + self.opt.eikonal_loss_weight * eikonal_loss
            loss_dict['eikonal'] = eikonal_loss.detach()

        if self.opt.env_sph_mode and self.opt.sdf_loss_weight > 0:
            surf_sdfs = outputs['surf_sdfs']
            sdf_loss = surf_sdfs.abs().mean()
            loss = loss + self.opt.sdf_loss_weight * sdf_loss
            loss_dict['sdf'] = sdf_loss.detach()
        # extra loss
        # pred_weights_sum = outputs['weights_sum'] + 1e-8
        # loss_ws = - 1e-1 * pred_weights_sum * torch.log(pred_weights_sum) # entropy to encourage weights_sum to be 0 or 1.
        # loss = loss + loss_ws.mean()

        return pred_rgb, gt_rgb, loss, loss_dict

    def eval_step(self, data, env_rot_radian=None):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]
        B, H, W, C = images.shape
        r_images = data['r_images'].reshape(B, -1, 3) if 'r_images' in data else None

        env_net_index = None
        if "env_net_indices" in data:
            env_net_indices = data["env_net_indices"].to(self.device)
            # assume batch size is 1
            assert env_net_indices.shape[0] == 1
            env_net_index = env_net_indices[0]
        
        if self.opt.set_env_net_index > 0:
            env_net_index = torch.tensor(self.opt.set_env_net_index).to(self.device)
        
        material = data['material'] if 'material' in data else None
        if self.opt.render_env_on_sphere: 
            material = {
                "roughness": self.opt.unwrap_roughness, # TODO make it opt
                "metallic": 1.0,
                "color": [self.opt.unwrap_color_intensity for i in range(3)] #[0.7, 0.7, 0.7]
            }  
        if self.opt.overwrite_materials:
            material['roughness'] = self.opt.unwrap_roughness     
            material['metallic'] = self.opt.unwrap_metallic
            material['color'] = self.opt.unwrap_color #[0.7, 0.7, 0.7]

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        # eval with fixed background color
        bg_color = 1 if self.opt.render_bg_color == 'white' else 0
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        if self.opt.debug:
            patch_h, patch_w = self.opt.debug_patch_h, self.opt.debug_patch_w # 497, 445 # 683, 314
            patch_size = self.opt.debug_patch_size
            rays_o = rays_o.reshape(B, H, W, -1)[:, patch_h:patch_h+patch_size, patch_w:patch_w+patch_size, :].reshape(B, -1, 3)
            rays_d = rays_d.reshape(B, H, W, -1)[:, patch_h:patch_h+patch_size, patch_w:patch_w+patch_size, :].reshape(B, -1, 3)
        
        use_specular_color = self.opt.color_net_start_iter == 0 or self.epoch > self.opt.color_net_start_iter
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, 
            get_normal_image=True, use_specular_color=use_specular_color, env_net_index=env_net_index, 
            epoch=self.epoch, material=material, r_images=r_images, env_rot_radian=env_rot_radian, **vars(self.opt))

        if self.opt.debug:
            return None
        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_normal = outputs['normal_image'].reshape(B, H, W, 3) if 'normal_image' in outputs else None
        visual_items = {}
        for k in self.opt.visual_items:
            image_k = k + '_image'
            if image_k in outputs:
                ch = outputs[image_k].shape[-1]
                visual_items[image_k] = outputs[image_k].reshape(B, H, W, ch)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, pred_normal, gt_rgb, loss, visual_items 

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False, get_normal_image=False):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, get_normal_image=get_normal_image, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)
        pred_normal = None
        if get_normal_image:
            pred_normal = outputs['normal_image'].reshape(-1, H, W, 3)

        return pred_rgb, pred_depth, pred_normal


    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func_density(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma

        def query_func_sdf(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sdf = self.model.density(pts.to(self.device))['sdf']
            return -sdf

        query_func = query_func_density
        if self.opt.use_sdf:
            query_func = query_func_sdf

        vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        # get a ref to error_map
        self.error_map = train_loader._data.error_map
        
        for epoch in range(self.epoch + 1, max_epochs + 1):

            self.epoch = epoch
            
            cfg_train_opt(self.opt, epoch, self.model, train_loader)

            if self.opt.use_neus_sdf:
                self.model.sdf_density.update_scale(self.opt.max_steps)
                if self.opt.rf_debug:
                    import IPython; IPython.embed()

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0 and \
                self.opt.ckpt_step > 0 and self.epoch % self.opt.ckpt_step == 0:
                self.save_checkpoint(full=True, best=False)

            # if self.workspace is not None:
            #     H, W = valid_loader._data.H, valid_loader._data.W
            #     intrinsics = valid_loader._data.intrinsics
            #     self.render_normal_image(intrinsics, H, W, "train_epoch_{}".format(self.epoch))

            if self.epoch % self.eval_interval == 0:
                debug = self.opt.debug
                self.opt.debug = False
                name = 'debug_frame' if debug else None
                # import IPython; IPython.embed()
                try:
                    torch.cuda.empty_cache()
                    d_only = self.opt.diffuse_only
                    self.opt.diffuse_only = False
                    self.evaluate_one_epoch(valid_loader, name=name)
                    self.opt.diffuse_only = d_only
                    self.opt.debug = debug
                    if debug:
                        self.evaluate_one_epoch(valid_loader)
                    self.save_checkpoint(full=True, best=True, both=self.opt.ckpt_step <= 0)
                except:
                    self.save_checkpoint(full=True, best=False, both=False)
                    print("OOM, exiting...")
                    exit()

            if self.opt.debug_stop_epoch > 0 and epoch > self.opt.debug_stop_epoch:
                exit()

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None, env_rot_degree_range=[]):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name, env_rot_degree_range)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        # with torch.no_grad(): # remove for normal image
        
        cfg_train_opt(self.opt, self.epoch, self.model)

        for i, data in enumerate(loader):
            
            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, preds_depth, _ = self.test_step(data, get_normal_image=True)

            if self.opt.color_space == 'linear' or self.opt.color_space == 'hdr':
                preds = linear_to_srgb(preds)

            pred = preds[0].detach().cpu().numpy().clip(0, 1)
            pred = (pred * 255).astype(np.uint8)

            pred_depth = preds_depth[0].detach().cpu().numpy()
            pred_depth = (pred_depth * 255).astype(np.uint8)

            if write_video:
                all_preds.append(pred)
                all_preds_depth.append(pred_depth)
            else:
                cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

            pbar.update(loader.batch_size)
        
        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")
    
    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        loader = iter(train_loader)

        # mark untrained grid
        if self.global_step == 0:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        for _ in range(step):
            
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
            
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss, _ = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            # print(f"beta={self.model.sdf_density.beta}, loss={loss}, grad={self.model.sdf_density.beta.grad}") # (2)
            # print("self.optimizer.param_groups[0]", self.optimizer.param_groups[0]) # (3)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        if self.model.use_sdf:
            outputs['beta'] = self.model.sdf_density.beta.abs()
        
        return outputs

    
    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1):
        
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        # with torch.no_grad(): # comment out to calculate normal = d_density / d_position
        with torch.cuda.amp.autocast(enabled=self.fp16):
            # here spp is used as perturb random seed! (but not perturb the first sample)
            preds, preds_depth, pred_normals = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp, get_normal_image=True)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            pred_normals = F.interpolate(pred_normals.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        pred_normals = pred_normals * 0.5 + 0.5 # [-1, 1] -> [0, 1]
        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)
            
        pred = preds[0].detach().cpu().numpy().clip(0, 1)
        pred_normal = pred_normals[0].detach().cpu().numpy()
        # pred_normal = pred_normal[..., [2,0,1]] # convert color space  # Crash on my windows laptop for unknown reason...
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
            "normal_image": pred_normal
        }

        return outputs

    def train_one_epoch(self, loader):
        start_time = time.time()
        start_train_str = f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f}"
        if self.model.use_sdf:
            if not self.opt.use_neus_sdf:
                start_train_str += f", beta={self.model.sdf_density.get_beta().item():.6f}"
            else:
                start_train_str += f", variance={self.model.sdf_density.get_variance().item():.6f}"
        start_train_str += " ..."
        self.log(start_train_str)

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            # update grid every 16 steps
            if self.model.cuda_ray and self.opt.update_extra_interval > 0 and \
                self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    if self.opt.extra_state_full_update:
                        self.model.reset_extra_state()
                        for i in range(16):
                            self.model.update_extra_state(full_update=True)
                        self.opt.extra_state_full_update = False
                    self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss, loss_dict = self.train_step(data, loader)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                pbar_description_str = f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})"
                for loss_dict_key in loss_dict:
                    pbar_description_str += f", {loss_dict_key}={loss_dict[loss_dict_key].item():.4f}"
                # if self.scheduler_update_every_step:
                #     pbar_description_str += f", lr={self.optimizer.param_groups[0]['lr']:.6f}"
                # if self.model.use_sdf:
                #     pbar_description_str += f", beta={self.model.sdf_density.beta.item():.6f}"
                pbar.set_description(pbar_description_str)
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        end_time = time.time()
        train_time = end_time - start_time
        self.log(f"==> Finished Epoch {self.epoch}. Time: {train_time:.4f}s")


    def render_normal_image(self, intrinsics, H, W, suffix):
        pose = get_one_pose(self.device, 2, np.pi / 2.0, np.pi)
        ray = get_rays(pose, intrinsics, H, W, -1)
        ray_o = ray['rays_o']
        ray_d = ray['rays_d']
        bg_color = 1
        model_status = self.model.training
        if model_status:
            self.model.eval()
        outputs = self.model.render(ray_o, ray_d, staged=True, bg_color=bg_color, perturb=False, get_normal_image=True, **vars(self.opt))
        pred_normals = outputs['normal_image'].reshape(H, W, 3)
        pred_normal = pred_normals * 0.5 + 0.5 # [-1, 1] -> [0, 1]
        pred_normal = pred_normal.detach().cpu().numpy()
        imageio.imwrite('{}/eval_{}.jpg'.format(self.workspace, suffix), (np.clip(pred_normal, 0.0, 1.0) * 255).astype(np.uint8))
        if model_status:
            self.model.train()

    def evaluate_one_epoch(self, loader, name=None, env_rot_degree_range=[]):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")
        use_env_rot = len(env_rot_degree_range) > 0
        if use_env_rot:
            degree_step = (env_rot_degree_range[1] - env_rot_degree_range[0]) / env_rot_degree_range[2]
            env_rot_degree_arr = torch.arange(env_rot_degree_range[0], env_rot_degree_range[1], degree_step)
        else:
            env_rot_degree_arr = torch.arange(0, 1) # only degree 0, i.e. no rotation
        num_rot_degree = env_rot_degree_arr.shape[0]

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None and self.epoch > 1:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        # with torch.no_grad(): # remove no grad for normal image
        self.local_step = 0

        for data in loader:    
            self.local_step += 1
            for env_rot_id, env_rot_degree in enumerate(env_rot_degree_arr):
                env_rot_radian = env_rot_degree * np.pi / 180.0
                if 'poses' in data:
                    poses = data['poses'].to(self.device)
                    rays = get_rays(poses, loader._data.intrinsics, loader._data.H, loader._data.W, loader._data.num_rays, loader._data.error_map, loader._data.opt.patch_size)
                    data['rays_o'] = rays['rays_o']
                    data['rays_d'] = rays['rays_d']
                    images = data['images'].to(self.device)
                    data['images'] = images
                    if 'r_images' in data:
                        r_images = data['r_images'].to(self.device)
                        data['r_images'] = r_images
                    else:
                        r_images = None
                else:
                    rays_o = data['rays_o'].to(self.device) # [B, N, 3]
                    rays_d = data['rays_d'].to(self.device) # [B, N, 3]

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    if self.opt.debug:
                        self.eval_step(data, env_rot_radian=env_rot_radian)
                        return
                    else:
                        preds, preds_depth, preds_normal, truths, loss, visual_items = self.eval_step(data, env_rot_radian=env_rot_radian)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_normal_list = [torch.zeros_like(preds_normal).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_normal_list, preds_normal)
                    preds_normal = torch.cat(preds_normal_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds, truths)

                    # save image
                    img_format = self.opt.img_format # 'png' 
                    nc_format = img_format if img_format != 'exr' else 'png'
                    rot_suffix = f'_rot{env_rot_id:03d}' if use_env_rot else ''
                    save_path = os.path.join(self.workspace, self.opt.val_folder_name, f'{name}', f'ep{self.epoch:04d}_{self.local_step:04d}_rgb{rot_suffix}.{img_format}')
                    save_path_depth = os.path.join(self.workspace, self.opt.val_folder_name, f'{name}', f'ep{self.epoch:04d}_{self.local_step:04d}_depth{rot_suffix}.{img_format}')
                    save_path_normal = os.path.join(self.workspace, self.opt.val_folder_name, f'{name}', f'ep{self.epoch:04d}_{self.local_step:04d}_normal{rot_suffix}.{nc_format}')
                    save_path_diffuse = os.path.join(self.workspace, self.opt.val_folder_name, f'{name}', f'ep{self.epoch:04d}_{self.local_step:04d}_diffuse{rot_suffix}.{img_format}')
                    save_path_specular = os.path.join(self.workspace, self.opt.val_folder_name, f'{name}', f'ep{self.epoch:04d}_{self.local_step:04d}_specular{rot_suffix}.{img_format}')
                    save_path_roughness = os.path.join(self.workspace, self.opt.val_folder_name, f'{name}', f'ep{self.epoch:04d}_{self.local_step:04d}_roughness{rot_suffix}.{nc_format}')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if self.opt.color_space == 'linear' or \
                    (self.opt.color_space == 'hdr' and img_format != 'exr'):
                        preds = linear_to_srgb(preds)
                        
                    pred = preds[0].detach().cpu().numpy()
                    if img_format != 'exr':
                        pred = (pred.clip(0, 1) * 255).astype(np.uint8)
                    imageio.imwrite(save_path, pred)

                    # pred_depth = preds_depth[0].detach().cpu().numpy()
                    # pred_depth = (pred_depth * 255).astype(np.uint8)
                    # cv2.imwrite(save_path_depth, pred_depth)

                    if preds_normal is not None:
                        pred_normal = preds_normal[0].detach().cpu().numpy() * 0.5 + 0.5
                        pred_normal = (pred_normal * 255).astype(np.uint8)
                        pred_normal = pred_normal[..., [2,0,1]] # convert color space yzx -> xyz
                        imageio.imwrite(save_path_normal, pred_normal)

                    if 'diffuse_image' in visual_items:
                        diffuse_image = visual_items['diffuse_image']
                        if self.opt.color_space == 'linear' or \
                        (self.opt.color_space == 'hdr' and self.opt.img_format != 'exr'):
                            diffuse_image = linear_to_srgb(diffuse_image)
                        diffuse_image = diffuse_image[0].detach().cpu().numpy()
                        if img_format != 'exr':
                            diffuse_image = (diffuse_image.clip(0, 1) * 255).astype(np.uint8)
                        imageio.imwrite(save_path_diffuse, diffuse_image)

                    
                    if 'specular_image' in visual_items:
                        specular_image = visual_items['specular_image']
                        if self.opt.color_space == 'linear' or \
                        (self.opt.color_space == 'hdr' and self.opt.img_format != 'exr'):
                            specular_image = linear_to_srgb(specular_image)
                        specular_image = specular_image[0].detach().cpu().numpy()
                        if img_format != 'exr':
                            specular_image = (specular_image.clip(0, 1) * 255).astype(np.uint8)
                        imageio.imwrite(save_path_specular, specular_image)

                    if 'roughness_image' in visual_items:
                        roughness_image = visual_items['roughness_image']
                        roughness_image = roughness_image[0].detach().cpu().numpy()
                        plt.figure()
                        plt.imshow(roughness_image, cmap='gray') #, vmax=1, vmin=0); 
                        plt.colorbar()
                        plt.savefig(save_path_roughness, dpi=200)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step / num_rot_degree
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None and self.epoch > 1:
            self.ema.restore()

        if self.model.use_sdf:
            if not self.model.opt.use_neus_sdf:
                self.log(f"beta={self.model.sdf_density.beta}")
            else:
                self.log(f"variance={self.model.sdf_density.variance}")
        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_env_net(self, env_nets, env_paths):
        assert len(env_nets) == len(env_paths)
        for i in range(len(env_nets)):
            state = {
                'model': env_nets[i].state_dict()
            }
            torch.save(state, env_paths[i]) 

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True, both=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best or both:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            # env_net_paths = []
            # if self.opt.env_sph_mode:
            #     for env_name in self.model.env_opt.env_images_names:
            #         env_net_paths.append(os.path.join(self.env_ckpt_path, f'env_{env_name}_ep{self.epoch:04d}.pth'))
            # else:
            #     env_net_paths.append(os.path.join(self.env_ckpt_path, f'env_net.pth'))

            if remove_old:
                self.stats["checkpoints"].append(file_path)
                # self.stats["env_checkpoints"].append(env_net_paths)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

                if len(self.stats["env_checkpoints"]) > self.max_keep_ckpt:
                    old_ckpts = self.stats["env_checkpoints"].pop(0)
                    for old_ckpt in old_ckpts:
                        if os.path.exists(old_ckpt):
                            os.remove(old_ckpt)

            # env_nets = self.model.env_nets if self.opt.env_sph_mode else [self.model.env_net]
            torch.save(state, file_path)
            # self.save_env_net(env_nets, env_net_paths)

        if best:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if 'density_grid' in state['model']:
                        del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
                    # env_nets = self.model.env_nets if self.opt.env_sph_mode else [self.model.env_net]
                    # env_net_paths = self.best_env_paths if self.opt.env_sph_mode else [self.best_env_path]
                    # self.save_env_net(env_nets, env_net_paths)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.loaded_from_ckpt = True
            self.log("[INFO] loaded model.")
            return

        if self.opt.swap_env_path != '' and self.opt.use_env_net:
            print(f"loading env from: {self.opt.swap_env_path}")
            state_dict = torch.load(self.opt.swap_env_path, map_location=self.device)
            env_state = state_dict['model']
            net_prefix = 'env_net.'
            env_state = {k: v for k, v in env_state.items() if k.startswith(net_prefix)}
            env_keys = [k for k in checkpoint_dict['model'].keys() if k.startswith(net_prefix)]
            for k in env_keys:
                if self.opt.split_diffuse_env:
                    checkpoint_dict['model']['diffuse_'+k] = checkpoint_dict['model'][k]
                del checkpoint_dict['model'][k]
            for k, v in env_state.items():
                checkpoint_dict['model'][k] = v
            # import IPython; IPython.embed()
            # self.model.env_net.load_state_dict(env_state)
        
        renv_prefix = 'renv_net.'
        renv_keys = [k for k in checkpoint_dict['model'].keys() if k.startswith(renv_prefix)]
        if len(renv_keys) > 0:
            self.load_renv = True

        model_state = self.model.state_dict()
        skip_optim = False
        for k in checkpoint_dict['model'].keys():
            if k in model_state:
                if model_state[k].shape != checkpoint_dict['model'][k].shape:
                    self.log(f"[WARN] shape mismatch: {k}, {model_state[k].shape} != {checkpoint_dict['model'][k].shape}")
                    self.log(f"[WARN] shape mismatch: extend ckpt shape to {model_state[k].shape}")
                    ori_param = checkpoint_dict['model'][k]
                    new_param = model_state[k]
                    new_param[:ori_param.shape[0]] = ori_param
                    checkpoint_dict['model'][k] = new_param
                    skip_optim = True

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.loaded_from_ckpt = True
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
        
        if model_only:
            return

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
            except:
                self.log("[WARN] ema loading fialed, reinit")
                self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.ema_decay)
            
        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict and not skip_optim:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
                # for g in self.optimizer.param_groups:
                #     g['capturable'] = True
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")

    def train_geometric_cue(self, opt):
        # Train SDF of a sphere
        bound = self.model.bound
        radius = bound * opt.scale
        center = torch.tensor([0, 0, 0], dtype=torch.float32).to(self.device)
        voxel_dims = torch.tensor([128, 128, 128], dtype=torch.int32).to(self.device)
        if self.loaded_from_ckpt:
            self.log("[INFO] Model loaded from checkpoint, skip training geometric cue.")
            return
        for _ in tqdm.tqdm(range(500), "Training geometric cue"):
            self.optimizer.zero_grad()
            coords = coordinates(voxel_dims - 1, self.device).float().t()
            coords = coords + torch.rand_like(coords) # [0, voxel_dims]
            coords = coords / voxel_dims * 2.0 - 1.0 # [-1, 1]
            coords = coords * bound
            sdf, sigma, _ = self.model.forward_geometry(coords)
            sdf = sdf.squeeze(-1)
            target_sdf = (coords - center).norm(dim=-1) - radius
            loss = torch.nn.functional.mse_loss(sdf, target_sdf)
            if loss.item() < 1e-10:
                break
            loss.backward()
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        print("Init loss after geom init (sphere)", loss.item())

def coordinates(voxel_dim, device: torch.device):
    nx, ny, nz = voxel_dim[0], voxel_dim[1], voxel_dim[2]
    x = torch.arange(0, nx, dtype=torch.long, device=device)
    y = torch.arange(0, ny, dtype=torch.long, device=device)
    z = torch.arange(0, nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z)

    return torch.stack((x.flatten(), y.flatten(), z.flatten()))

def cfg_train_opt(opt, epoch, model=None, dataloader=None):
    if opt.relsdf_loss_weight > 0:
        opt.relsdf_loss = True
        if  opt.relsdf_loss_start_iter > epoch:
            opt.relsdf_loss = False

    if opt.dist_bound_weight > 0:
        opt.dist_bound = True
        if  opt.dist_bound_start_iter > epoch:
            opt.dist_bound = False

    if opt.backsdf_loss_weight > 0:
        opt.backsdf_loss = True
        if  opt.backsdf_loss_start_iter > epoch:
            opt.backsdf_loss = False

    if opt.cauchy_loss_weight > 0:
        opt.cauchy_loss = True
        if  opt.cauchy_loss_start_iter > epoch:
            opt.cauchy_loss = False

    if opt.eikonal_loss_weight > 0:
        opt.eikonal_loss = True
        if  opt.eikonal_loss_start_iter > epoch:
            opt.eikonal_loss = False
    
    if opt.orientation_loss_weight > 0:
        opt.orientation_loss = True
        if  opt.orientation_loss_start_iter > epoch:
            opt.orientation_loss = False
    
    if opt.mask_loss_weight > 0:
        opt.mask_loss = True
        if  opt.mask_loss_start_iter > epoch:
            opt.mask_loss = False

    if opt.error_bound_start_iter > epoch:
        opt.error_bound_sample = False
    elif opt.error_bound_start_iter > 0:
        opt.error_bound_sample = True

    if opt.color_net_start_iter > epoch:
        opt.diffuse_only = True
    else:
        opt.diffuse_only = False

    if opt.indir_ref_start_iter > 0 and opt.indir_ref_start_iter <= epoch:
        opt.indir_ref = True
        if opt.grad_rays_start_iter > 0 and epoch - opt.indir_ref_start_iter > opt.grad_rays_start_iter:
            opt.grad_rays = True
    else:
        opt.indir_ref = False

    if opt.normal_anneal_iters > 0:
        opt.normal_anneal_ratio = min(epoch / opt.normal_anneal_iters, 1)
        
    if len(opt.enabled_levels_sched) > 0 and opt.enabled_levels < opt.num_levels:
        base_level, s_start, s_iters = opt.enabled_levels_sched
        final_level = opt.num_levels
        iters = epoch - s_start
        # linearly annealing from base_level to final_level for # s_iters epochs
        if iters >= 0:
            opt.enabled_levels = base_level + (final_level - base_level) * min(iters / s_iters, 1)
            opt.enabled_levels = int(opt.enabled_levels)
        else:
            opt.enabled_levels = base_level

    if len(opt.relsdf_loss_weight_sched) > 0:
        # exponentially annealing from opt.relsdf_loss_weight_start to opt.relsdf_loss_weight_end for # opt.relsdf_loss_weight_sched epochs
        w_start, w_end, s_start, s_iters, s_stop = opt.relsdf_loss_weight_sched
        iters = epoch - s_start
        if s_stop > 0 and s_stop <= iters:
            opt.relsdf_loss_weight = 0
        elif iters >= 0:
            opt.relsdf_loss_weight = w_start * (w_end / w_start) ** min(iters / s_iters, 1)
            # opt.relsdf_loss_weight = w_start + (w_end - w_start) * min(iters / s_iters, 1)

    if len(opt.cauchy_loss_weight_sched) > 0:
        # exponentially annealing from opt.cauchy_loss_weight_start to opt.cauchy_loss_weight_end for # opt.cauchy_loss_weight_sched epochs
        w_start, w_end, s_start, s_iters, s_stop = opt.cauchy_loss_weight_sched
        iters = epoch - s_start
        if s_stop > 0 and s_stop <= iters:
            opt.cauchy_loss_weight = 0
        elif iters >= 0:
            opt.cauchy_loss_weight = w_start * (w_end / w_start) ** min(iters / s_iters, 1)
            # opt.cauchy_loss_weight = w_start + (w_end - w_start) * min(iters / s_iters, 1)
    
    if len(opt.backsdf_loss_weight_sched) > 0:
        # exponentially annealing from opt.backsdf_loss_weight_start to opt.backsdf_loss_weight_end for # opt.backsdf_loss_weight_sched epochs
        w_start, w_end, s_start, s_iters, s_stop = opt.backsdf_loss_weight_sched
        iters = epoch - s_start
        if s_stop > 0 and s_stop <= iters:
            opt.backsdf_loss_weight = 0
        elif iters >= 0:
            opt.backsdf_loss_weight = w_start * (w_end / w_start) ** min(iters / s_iters, 1)
            # opt.backsdf_loss_weight = w_start + (w_end - w_start) * min(iters / s_iters, 1)
    
    if len(opt.eikonal_loss_weight_sched) > 0:
        # exponentially annealing from opt.eikonal_loss_weight_start to opt.eikonal_loss_weight_end for # opt.eikonal_loss_weight_sched epochs
        w_start, w_end, s_start, s_iters, s_stop = opt.eikonal_loss_weight_sched
        iters = epoch - s_start
        if s_stop > 0 and s_stop <= iters:
            opt.eikonal_loss_weight = 0
        elif iters >= 0:
            opt.eikonal_loss_weight = w_start * (w_end / w_start) ** min(iters / s_iters, 1)
            # opt.eikonal_loss_weight = w_start + (w_end - w_start) * min(iters / s_iters, 1)

    if len(opt.orientation_loss_weight_sched) > 0:
        # exponentially annealing from opt.orientation_loss_weight_start to opt.orientation_loss_weight_end for # opt.orientation_loss_weight_sched epochs
        w_start, w_end, s_start, s_iters, s_stop = opt.orientation_loss_weight_sched
        iters = epoch - s_start
        if s_stop > 0 and s_stop <= iters:
            opt.orientation_loss_weight = 0
        elif iters >= 0:
            opt.orientation_loss_weight = w_start * (w_end / w_start) ** min(iters / s_iters, 1)
            # opt.orientation_loss_weight = w_start + (w_end - w_start) * min(iters / s_iters, 1)


    if opt.use_sdf and len(opt.beta_min_sched) > 0:
        bm_start, bm_end, bm_iters = opt.beta_min_sched
        model.sdf_density.beta_min = bm_start * (bm_end / bm_start) ** min(epoch / bm_iters, 1)
        # model.sdf_density.beta_min = bm_start + (bm_end - bm_start) * min(epoch / bm_iters, 1)

    if opt.cuda_ray and len(opt.early_stop_steps_sched) > 0:
        steps, iters = opt.early_stop_steps_sched[:2]
        while epoch >= iters:
            opt.early_stop_steps = steps
            opt.early_stop_steps_sched = opt.early_stop_steps_sched[2:]
            if len(opt.early_stop_steps_sched) > 0:
                steps, iters = opt.early_stop_steps_sched[:2]
            else:
                iters = 1e10
            if hasattr(model, 'mean_count'):
                model.mean_count = -1
        
    if opt.cuda_ray and len(opt.max_steps_sched) > 0:
        steps, iters = opt.max_steps_sched[:2]
        while epoch >= iters:
            opt.max_steps = steps
            opt.max_steps_sched = opt.max_steps_sched[2:]
            if len(opt.max_steps_sched) > 0:
                steps, iters = opt.max_steps_sched[:2]
            else:
                iters = 1e10

    if opt.cuda_ray and len(opt.num_rays_sched) > 0:
        steps, iters = opt.num_rays_sched[:2]
        while epoch >= iters:
            opt.num_rays = steps
            if hasattr(dataloader, '_data'):
                dataloader._data.num_rays = steps
            opt.num_rays_sched = opt.num_rays_sched[2:]
            if len(opt.num_rays_sched) > 0:
                steps, iters = opt.num_rays_sched[:2]
            else:
                iters = 1e10
            if hasattr(model, 'mean_count'):
                model.mean_count = -1

    if opt.error_map and opt.error_map_start_iter > 0:
        if epoch >= opt.error_map_start_iter:
            opt.error_map_start_iter = -1

    if opt.update_extra_before >= 0:
        if epoch >= opt.update_extra_before:
            opt.update_extra_interval = -1