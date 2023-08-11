import math
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import raymarching
from .utils import custom_meshgrid, rot_theta

import nerf.render_func as render_func

import sys
import copy

SQRT3 = 3**0.5

def reflect_dir(w_o, normals):
    """Reflect view directions about normals.

    The reflection of a vector v about a unit vector n is a vector u such that
    dot(v, n) = dot(u, n), and dot(u, u) = dot(v, v). The solution to these two
    equations is u = 2 dot(n, v) n - v.

    Caution: 
        * This function assumes that the w_o and normals are unit vectors.
        * w_o is direction from the surface to the camera, *unlike* ray_dir in NeRF.

    Args:
        viewdirs: [..., 3] array of view directions.
        normals: [..., 3] array of normal directions (assumed to be unit vectors).

    Returns:
        [..., 3] array of reflection directions, surf to light
    """
    w_r = 2 * torch.sum(w_o * normals, dim=-1, keepdim=True) * normals - w_o 
    return w_r


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


class NeRFRenderer(nn.Module):
    def __init__(self,
                 bound=1,
                 cuda_ray=False,
                 density_scale=1, # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
                 min_near=0.2,
                 density_thresh=0.01,
                 bg_radius=-1,
                 use_sdf = True,
                 opt=None,
                 env_opt=None,
                 **kwargs
                 ):
        super().__init__()

        self.opt = opt
        self.env_opt = env_opt
        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius # radius of the background sphere.

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        if len(opt.marching_aabb) == 6:
            aabb_train = torch.FloatTensor(opt.marching_aabb) * opt.scale
            aabb_train = aabb_train.clamp(min=-bound, max=bound)
            print(f"using marching_aabb={aabb_train}")
        else:
            aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        if opt.obj_aabb is not None:
            obj_aabb = torch.FloatTensor(opt.obj_aabb) * opt.scale
            obj_aabb = obj_aabb.clamp(min=-bound, max=bound)
            print(f"using obj_aabb={obj_aabb}")
            self.register_buffer('obj_aabb', obj_aabb, persistent=False)
        else:
            self.obj_aabb = None

        self.use_sdf = use_sdf
        self.use_normal_with_mlp = self.opt.normal_with_mlp
        self.use_reflected_dir = self.opt.use_reflected_dir
        self.use_n_dot_viewdir = self.opt.use_n_dot_viewdir
        print(f"use_sdf={self.use_sdf}, use_normal_with_mlp={self.use_normal_with_mlp}, use_reflected_dir={self.use_reflected_dir}, use_n_dot_viewdir={self.use_n_dot_viewdir}")

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        if cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0
    
    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not self.cuda_ray:
            return 
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0
    
    def get_grad(self, y, x):
        retain_graph = self.training
        return torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y), retain_graph=retain_graph, create_graph=retain_graph)[0]

    def get_color_mlp_extra_params(self, normals, dirs, roughness=0, env_rot_radian=None):
        if normals is None:
            return None, None, None, None
        normals_enc = None
        if self.use_normal_with_mlp:
            normals_enc = self.encoder_normal(normals)
        
        # Assume dirs is normalized
        w_o = - dirs # unit vector pointing from a point in space to the camera
        w_r_enc = None
        if self.use_reflected_dir and not self.opt.diffuse_only:
            w_r = reflect_dir(w_o, normals)
            # TODO: rotate w_r
            if env_rot_radian is not None:
                w_r = w_r @ torch.from_numpy(rot_theta(env_rot_radian)[:3, :3]).float().to(w_r.device)
            w_r_enc = self.encoder_refdir(w_r, roughness=roughness)
        if w_r_enc is not None:
            w_r_enc = w_r_enc * self.opt.light_intensity_scale
        n_dot_w_o = None
        if self.use_n_dot_viewdir:
            n_dot_w_o = torch.sum(normals * w_o, dim=-1, keepdim=True)
        n_env_enc = None
        if self.opt.diffuse_with_env:
            # TODO: rotate normals
            if env_rot_radian is not None:
                normals = normals @ torch.from_numpy(rot_theta(env_rot_radian)[:3, :3]).float().to(normals.device)
            if self.opt.split_diffuse_env:
                n_env_enc = self.diffuse_encoder_refdir(normals, roughness=self.opt.diffuse_kappa_inv)
            else:
                n_env_enc = self.encoder_refdir(normals, roughness=self.opt.diffuse_kappa_inv)
            if n_env_enc is not None:
                n_env_enc = n_env_enc * self.opt.light_intensity_scale

        return normals_enc, w_r_enc, n_dot_w_o, n_env_enc

    def compute_normal(self, sdf_or_sigma, xyzs, get_eikonal_sdf_gradient=False):
        eikonal_sdf_gradient = None
        d_outpus = torch.ones_like(sdf_or_sigma)
        if self.use_sdf:
            normals = torch.autograd.grad(outputs=sdf_or_sigma, inputs=xyzs, grad_outputs=d_outpus, retain_graph=True, create_graph=True)[0]
        else:
            normals = - torch.autograd.grad(outputs=sdf_or_sigma, inputs=xyzs, grad_outputs=d_outpus, retain_graph=True, create_graph=True)[0]
        if get_eikonal_sdf_gradient:
            eikonal_sdf_gradient = normals # for Eikonal loss
        normals = normals.detach() if self.opt.detach_normal else normals
        normals = F.normalize(normals, dim=-1, eps=1e-10)
        if self.opt.normal_anneal_ratio < 1:
            normals = normals * self.opt.normal_anneal_ratio + \
                (1 - self.opt.normal_anneal_ratio) * F.normalize(xyzs.detach(), dim=-1, eps=1e-10)
            normals = F.normalize(normals, dim=-1, eps=1e-10)
        
        return normals, eikonal_sdf_gradient

    @torch.no_grad()
    def mark_untrained_grid(self, poses, intrinsic, S=64):
        # poses: [B, 4, 4]
        # intrinsic: [3, 3]

        if not self.cuda_ray:
            return
        
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        B = poses.shape[0]
        
        fx, fy, cx, cy = intrinsic
        
        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        count = torch.zeros_like(self.density_grid)
        poses = poses.to(count.device)

        # 5-level loop, forgive me...

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0) # [1, N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                        # split batch to avoid OOM
                        head = 0
                        while head < B:
                            tail = min(head + S, B)

                            # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed for batched matmul, so the final form is without transpose.)
                            cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3] # [S, N, 3]
                            
                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] > 0 # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1) # [N]

                            # update count 
                            count[cas, indices] += mask
                            head += S
    
        # mark untrained grid as -1
        self.density_grid[count == 0] = -1

        print(f'[mark untrained grid] {(count == 0).sum()} from {self.grid_size ** 3 * self.cascade}')

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128, full_update=False):
        # call before each epoch to update extra states.
        # print("update extra state start", file=sys.stderr)

        if not self.cuda_ray:
            return 
        
        ### update density grid

        tmp_grid = - torch.ones_like(self.density_grid)
        
        # full update.
        if self.iter_density < 16 or full_update:
        #if True:
            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

            for xs in X:
                for ys in Y:
                    for zs in Z:
                        
                        # construct points
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                        indices = raymarching.morton3D(coords).long() # [N]
                        xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                        # cascading
                        for cas in range(self.cascade):
                            bound = min(2 ** cas, self.bound)
                            half_grid_size = bound / self.grid_size
                            # scale to current cascade's resolution
                            cas_xyzs = xyzs * (bound - half_grid_size)
                            # add noise in [-hgs, hgs]
                            cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                            # query density
                            sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                            sigmas *= self.density_scale
                            # assign 
                            tmp_grid[cas, indices] = sigmas

        # partial update (half the computation)
        # TODO: why no need of maxpool ?
        else:
            N = self.grid_size ** 3 // 4 # H * H * H / 4
            for cas in range(self.cascade):
                # random sample some positions
                coords = torch.randint(0, self.grid_size, (N, 3), device=self.density_bitfield.device) # [N, 3], in [0, 128)
                indices = raymarching.morton3D(coords).long() # [N]
                # random sample occupied positions
                occ_indices = torch.nonzero(self.density_grid[cas] > 0).squeeze(-1) # [Nz]
                rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=self.density_bitfield.device)
                occ_indices = occ_indices[rand_mask] # [Nz] --> [N], allow for duplication
                occ_coords = raymarching.morton3D_invert(occ_indices) # [N, 3]
                # concat
                indices = torch.cat([indices, occ_indices], dim=0)
                coords = torch.cat([coords, occ_coords], dim=0)
                # same below
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]
                bound = min(2 ** cas, self.bound)
                half_grid_size = bound / self.grid_size
                # scale to current cascade's resolution
                cas_xyzs = xyzs * (bound - half_grid_size)
                # add noise in [-hgs, hgs]
                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                # query density
                sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                sigmas *= self.density_scale
                # assign 
                tmp_grid[cas, indices] = sigmas

        ## max-pool on tmp_grid for less aggressive culling [No significant improvement...]
        # invalid_mask = tmp_grid < 0
        # tmp_grid = F.max_pool3d(tmp_grid.view(self.cascade, 1, self.grid_size, self.grid_size, self.grid_size), kernel_size=3, stride=1, padding=1).view(self.cascade, -1)
        # tmp_grid[invalid_mask] = -1

        # ema update
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item() # -1 regions are viewed as 0 density.
        #self.mean_density = torch.mean(self.density_grid[self.density_grid > 0]).item() # do not count -1 regions
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0
        # print("update extra state end", file=sys.stderr)

        #print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > 0.01).sum() / (128**3 * self.cascade):.3f} | [step counter] mean={self.mean_count}')


    def render(self, rays_o, rays_d, staged=False, max_ray_batch=4096, get_normal_image=False, use_specular_color=True, env_net_index=None, material=None, r_images=None, env_rot_radian=None, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        if self.cuda_ray:
            _run = render_func.run_cuda
        else:
            _run = render_func.run
        
        if self.opt.error_bound_sample:
            _run = render_func.run_volsdf

        if self.opt.env_sph_mode or self.opt.render_env_on_sphere:
            _run = render_func.run_sph

        B, N = rays_o.shape[:2]
        device = rays_o.device

        kwargs['material'] = material
        
        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)
            normal = torch.empty((B, N, 3), device=device)
            if 'diffuse' in self.opt.visual_items:
                diffuse = torch.empty((B, N, 3), device=device)
            if 'specular' in self.opt.visual_items:
                specular = torch.empty((B, N, 3), device=device)
            if 'roughness' in self.opt.visual_items:
                roughness = torch.empty((B, N, 1), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    if r_images is not None:
                        kwargs['r_images'] = r_images[b:b+1, head:tail]
                    results_ = _run(self, rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], get_normal_image=get_normal_image, env_net_index=env_net_index, env_rot_radian=env_rot_radian, **kwargs)
                    depth[b:b+1, head:tail] = results_['depth']
                    image[b:b+1, head:tail] = results_['image']
                    if 'normal_image' in results_ and results_['normal_image'] is not None:
                        normal[b:b+1, head:tail] = results_['normal_image']
                    if 'diffuse' in self.opt.visual_items:
                        diffuse[b:b+1, head:tail] = results_['diffuse_image']
                    if 'specular' in self.opt.visual_items:
                        specular[b:b+1, head:tail] = results_['specular_image']
                    if 'roughness' in self.opt.visual_items:
                        roughness[b:b+1, head:tail] = results_['roughness_image']
                    head += max_ray_batch
            
            results = {}
            results['depth'] = depth
            results['image'] = image
            results['normal_image'] = normal
            if 'diffuse' in self.opt.visual_items:
                results['diffuse_image'] = diffuse
            if 'specular' in self.opt.visual_items:
                results['specular_image'] = specular
            if 'roughness' in self.opt.visual_items:
                results['roughness_image'] = roughness

        else:
            _rays_o, _rays_d = rays_o, rays_d
            if self.opt.max_ray_batch_cuda > 0:
                num_iters = (N - 1) // self.opt.max_ray_batch_cuda + 1
                _results = {}
            else:
                num_iters = 1
            for i in range(num_iters):
                if num_iters > 1:
                    rays_o = _rays_o[:, i*self.opt.max_ray_batch_cuda:(i+1)*self.opt.max_ray_batch_cuda]
                    rays_d = _rays_d[:, i*self.opt.max_ray_batch_cuda:(i+1)*self.opt.max_ray_batch_cuda]
                if not self.opt.indir_ref:
                    results = _run(self, rays_o, rays_d, get_normal_image=get_normal_image, use_specular_color=use_specular_color, env_net_index=env_net_index, r_images=r_images, env_rot_radian=env_rot_radian, **kwargs)
                else:
                    dt =  2 * SQRT3 / self.opt.indir_max_steps # diagnal
                    # first pass: geometry only
                    geo_res = _run(self, rays_o, rays_d, get_normal_image=get_normal_image, main_pass=False, geometry_only=True, env_rot_radian=env_rot_radian, **kwargs)
                    normals = geo_res['normal_image'] # normalized
                    depth = geo_res['depth'].squeeze() - dt
                    weights_sum = geo_res['weights_sum'].squeeze()
                    # roughness = results['roughness_image']
                    # TODO mask out
                    
                    # second pass: self reflection
                    # roughness_thresh = 0.08
                    with torch.no_grad():
                        ref_mask = (depth != 0) * (weights_sum > 0.9)
                        ray_mask = (depth != 0) * (weights_sum > 0.3)
                        # ray_mask = ref_mask
                    ref_o = rays_o + depth[None,...,None] * rays_d # [1, R, 3]
                    ref_d = reflect_dir(-rays_d, normals)
                    # import IPython; IPython.embed() 
                    if self.obj_aabb is not None:
                        bnd_mask = (ref_o[0] > self.obj_aabb[:3]).all(-1) * (ref_o[0] < self.obj_aabb[3:]).all(-1)
                        ref_mask = ref_mask * bnd_mask
                    
                    _bg_color, _early_stop_steps = kwargs['bg_color'], kwargs['early_stop_steps']
                    _max_steps = kwargs['max_steps']
                    if _bg_color is None:
                        _bg_color = 0
                    if 'sphere_bg' in geo_res:
                        _bg_color = geo_res['sphere_bg']
                    _min_near = getattr(self, 'min_near', 0.2)
                    self.min_near = dt * 2
                    kwargs['bg_color'], kwargs['early_stop_steps'], kwargs['max_steps'] = \
                        0, self.opt.indir_early_stop_steps, self.opt.indir_max_steps
                    kwargs['force_all_rays'] = True
                    ref_res = _run(self, ref_o[:, ref_mask, :], ref_d[:, ref_mask, :], get_normal_image=get_normal_image, use_specular_color=use_specular_color, 
                        env_net_index=env_net_index, main_pass=False, grad_ray=self.opt.grad_rays, bg_sphere=False, env_rot_radian=env_rot_radian, **kwargs)
                    
                    ref_image, ref_weights_sum = ref_res['image'], ref_res['weights_sum']
                    ref_image = torch.cat([ref_image, ref_weights_sum[...,None]], -1)

                    # if not self.opt.grad_rays:
                    #     ref_image = ref_image.detach()
                    # ref_image = ref_image.new_zeros(*normals.shape[:-1], 4).masked_scatter(ref_mask[None, :, None], ref_image)
                    
                    # ref_image = torch.zeros_like(normals).masked_scatter(ref_mask[None, :, None], ref_res['image'])
                    # results['specular_image'] = ref_res['specular_image']
                    # results['image'] = ref_res['image']
                    # results['normal_image'] = ref_res['normal_image']
                    # results['ref_image'] = ref_image

                    ref2ray_mask = ref_mask[ray_mask]
                    ref_image = ref_image.new_zeros(1, ref2ray_mask.shape[0], 4).masked_scatter(ref2ray_mask[None, :, None], ref_image)

                    # TODO: third pass        
                    kwargs['early_stop_steps'] = _early_stop_steps
                    kwargs['max_steps'] = _max_steps
                    self.min_near = _min_near
                    results = _run(self, rays_o[:, ray_mask, :], rays_d[:, ray_mask, :], get_normal_image=get_normal_image, use_specular_color=use_specular_color, 
                        env_net_index=env_net_index, main_pass=True, r_images=ref_image, bg_sphere=False, env_rot_radian=env_rot_radian, **kwargs)
                    
                    kwargs['bg_color'] = _bg_color
                    
                    results['normal_image'] = normals
                    results['depth'] = depth[None, :]
                    k_fields = ['image', 'specular_image', 'diffuse_image', 'roughness_image']
                    for k in k_fields:
                        if k in results:
                            vdim = results[k].shape[-1]
                            v = normals.new_zeros(*normals.shape[:-1], vdim).masked_scatter(ray_mask[None, :, None], results[k])
                            results[k] = v
                    image = torch.zeros_like(normals) + _bg_color
                    weights_sum = normals.new_zeros(*normals.shape[:-1], 1).masked_scatter(ray_mask[None, :, None], results['weights_sum'][...,None])
                    # weights_sum = geo_res['weights_sum'][..., None]
                    results['image'] = image * (1 - weights_sum) + results['image']
                    results['weights_sum'] = weights_sum.squeeze(-1)
                if num_iters > 1:
                    for k, v in results.items():
                        if k not in _results:
                            _results[k] = [v]
                        else:
                            _results[k].append(v)
            
            if num_iters > 1:
                results = {}
                for k, v in _results.items():
                    if k in ['diffuse_image', 'specular_image', 'roughness_image']:
                        results[k] = torch.cat(v, 0)
                    else:
                        results[k] = torch.cat(v, 1)
                            
        if 'weights_sum' in results and get_normal_image:
           results['normal_image'] = results['normal_image'] * (results['weights_sum'][...,None]) +  (1 - results['weights_sum'][...,None])
        return results