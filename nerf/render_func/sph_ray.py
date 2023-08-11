import torch 

from typing import Optional
import torch
import torch.nn.functional as F

import torch.nn as nn
from torch import Tensor

import raymarching
from .utils import curve_plot
import numpy as np

@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)

@torch.jit.script
def get_sphere_intersections(rays_o: Tensor, rays_d: Tensor, r: float=1.0):
    '''
    Ray-sphere Intersection: https://en.wikipedia.org/wiki/Line-sphere_intersection
    Input: n_rays x 3 ; n_rays x 3
    Output: n_rays x 1, n_rays x 1 (near and far)
    '''
    ray_cam_dot = torch.bmm(rays_d.view(-1, 1, 3),
                            rays_o.view(-1, 3, 1)).squeeze(-1)
    nabla = ray_cam_dot ** 2 - (rays_o.norm(2, 1, keepdim=True) ** 2 - r ** 2)
    nabla_sqrt = torch.sqrt(nabla.clamp_min(0.0))
    near = -ray_cam_dot - nabla_sqrt
    far = -ray_cam_dot + nabla_sqrt
    mask = (nabla >= -1e-4)[..., 0]
    return near, far, mask

def run_sph(model, rays_o, rays_d, dt_gamma=0, bg_color=None, perturb=False, num_step=12, step_size=0.002, get_normal_image=False, use_specular_color=True, env_net_index=None, material=None, r_images=None, **kwargs):
    # rays_o, rays_d: [B, N, 3], assumes B == 1
    # return: image: [B, N, 3], depth: [B, N]
    self = model

    prefix = rays_o.shape[:-1]
    rays_o = rays_o.contiguous().view(-1, 3)
    rays_d = rays_d.contiguous().view(-1, 3)

    debug = self.opt.debug
    use_backsdf_loss = self.opt.backsdf_loss
    use_eikonal_loss = self.opt.eikonal_loss
    use_normal_with_mlp = self.use_normal_with_mlp
    use_sdf_sigma_grad = debug or use_backsdf_loss or use_eikonal_loss or use_normal_with_mlp or self.use_n_dot_viewdir or self.use_reflected_dir

    radius = self.opt.env_sph_radius #* self.opt.scale

    N = rays_o.shape[0] # N = B * N, in fact
    device = rays_o.device

    bg_color = (torch.zeros(N, 3, device=device) + bg_color).reshape(N, 3)
    # get near and far
    nears, fars, mask = get_sphere_intersections(rays_o, rays_d, radius) # [N, 1], [N, 1], [N, 1]
    if not mask.any():
        results = {
            'image': bg_color.reshape(*prefix, 3), 
            'depth': torch.zeros_like(nears).reshape(*prefix),
            'normal_image': torch.zeros_like(bg_color).reshape(*prefix, 3) if get_normal_image else None,
            'diffuse_image': bg_color.reshape(*prefix, 3) if 'diffuse' in self.opt.visual_items else None,
            'specular_image': bg_color.reshape(*prefix, 3) if 'specular' in self.opt.visual_items else None,
            'roughness_image': torch.zeros_like(bg_color[...,:1]).reshape(*prefix, 1) if 'roughness' in self.opt.visual_items else None,
            'empty': True 
        }
        return results

    nears_valid = nears[mask] # [M, 1]

    z_radius = step_size * (num_step-1) / 2
    z_vals = torch.linspace(-z_radius, z_radius, num_step, device=device)[None, :] # [1, S]
    z_vals = z_vals + nears_valid # [M, S]

    if perturb:
        z_vals = z_vals + (torch.rand_like(z_vals) - 0.5) * step_size # [M, S]
    
    dirs = rays_d[mask, None, :] # [M, 1, 3]
    xyzs = rays_o[mask, None, :] + dirs * z_vals[:, :, None] # [M, S, 3]

    results = {}
        
    if use_sdf_sigma_grad or get_normal_image:
        xyzs.requires_grad = True
        
    sdfs, sigmas, geo_feats, normals, eikonal_sdf_gradients = self.forward_sigma(xyzs, use_sdf_sigma_grad=use_sdf_sigma_grad, material=material)

    roughness = getattr(self, 'roughness', self.opt.default_roughness)

    with torch.set_grad_enabled(self.training):
        sigmas = self.density_scale * sigmas
        
        normals_enc, w_r_enc, n_dot_w_o, n_env_enc = self.get_color_mlp_extra_params(normals, dirs, roughness)

        if self.opt.train_renv:
            r_images = r_images[0, mask, None, :].expand(-1, num_step, -1)
        
        if not self.training:
            normals_enc = normals_enc.detach()
        rgbs = self.forward_color(geo_feats, dirs, normals_enc, w_r_enc, n_dot_w_o, use_specular_color, env_net_index, n_env_enc=n_env_enc, r_images=r_images, roughness=roughness)

        #print(f'valid RGB query ratio: {mask.sum().item() / mask.shape[0]} (total = {mask.sum().item()})')
        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        deltas = torch.cat([deltas, step_size * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * sigmas.squeeze(-1)) # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]

        weights_sum = weights.sum(dim=-1, keepdim=True) # [N, 1]

    # calculate depth 
    ori_z_vals = ((z_vals - nears_valid) / (fars.max() - nears_valid)).clamp(0, 1)
    depth = torch.sum(weights * ori_z_vals, dim=-1)
    # unmasking
    depth = torch.zeros_like(nears).masked_scatter_(mask[...,None], depth)

    # calculate color
    image_m = torch.sum(weights[..., None] * rgbs, dim=-2) 
    image_m = image_m + (1 - weights_sum) * bg_color[mask]
    image = bg_color.masked_scatter(mask[...,None], image_m)

    if not self.training and get_normal_image:
        normal_image = torch.sum(weights[..., None] * normals.detach(), dim=1)
        normal_image = F.normalize(normal_image, dim=-1)
        normal_image = torch.zeros(N, 3, device=device).masked_scatter_(mask[...,None], normal_image).reshape(*prefix, 3)
    else:
        normal_image = None

    if self.opt.use_diffuse:
        if 'diffuse' in self.opt.visual_items:
            c_diffuse = self.c_diffuse
            diffuse_image = torch.sum(weights[..., None] * c_diffuse, dim=-2)
            diffuse_image = diffuse_image + (1 - weights_sum) * bg_color[mask]
            diffuse_image = bg_color.masked_scatter(mask[...,None], diffuse_image)
            results['diffuse_image'] = diffuse_image.reshape(*prefix, 3)

        if 'specular' in self.opt.visual_items:
            c_specular = self.c_specular
            specular_image = torch.sum(weights[..., None] * c_specular, dim=-2)
            specular_image = specular_image + (1 - weights_sum) * bg_color[mask]
            specular_image = bg_color.masked_scatter(mask[...,None], specular_image)
            results['specular_image'] = specular_image.reshape(*prefix, 3)
    
    if getattr(self, 'roughness', None) is not None and 'roughness' in self.opt.visual_items:
        roughness_image = torch.sum(weights[..., None] * roughness, dim=-2)
        roughness_image = torch.zeros(N, 1, device=device).masked_scatter_(mask[...,None], roughness_image)
        results['roughness_image'] = roughness_image.reshape(*prefix, 1)


    results['weights_sum'] = torch.zeros_like(nears).masked_scatter_(mask[...,None], weights_sum)
    results['sigmas'] = sigmas
    results['sdfs'] = sdfs
    if use_eikonal_loss:
        results['sdf_gradients'] = eikonal_sdf_gradients

    results['depth'] = depth.reshape(*prefix)
    results['image'] = image.reshape(*prefix, 3)
    results['normal_image'] = normal_image

    if self.training and self.opt.sdf_loss_weight > 0:
        surf_xyzs = rays_o[mask, None, :] + dirs * nears_valid[:, :, None] # [M, 1, 3]
        surf_sdfs, _, _, _, _ = self.forward_sigma(surf_xyzs, use_sdf_sigma_grad=False, material=material)
        results['surf_sdfs'] = surf_sdfs

    if use_backsdf_loss:
        # now compute the relsdf
        if not self.use_sdf:
            sdfs = -sigmas

        relsdf = sdfs[...,1:] - sdfs[..., :-1] # sdfs is decreasing along the ray, so relsdf should be negative
        # next is to compute the cos, make sure xyz_grads is normalized
        cos = (dirs * normals.detach()).sum(-1)[..., :-1] # this is cos of \pi/2 - \theta
        # now we can compute the approximated relsdf
        # est_relsdf = deltas[..., :-1] * cos # in most cases, this should be negative number
        results['relsdf'] = relsdf
        # results['est_relsdf'] = est_relsdf
        results['sdf_weights'] = weights[..., :-1]
        results['sdf_dist'] = deltas[..., :-1]

        # results['weights'] = weights
    if debug and not self.training:
        import os
        workspace = self.opt.workspace
        epoch = kwargs.get('epoch', 0)
        debug_path = os.path.join(workspace, 'debug', f'ep{epoch}')
        os.makedirs(debug_path, exist_ok=True)
        ray_o = rays_o[0].view(-1, 3)
        # ray_idx_map = torch.ones_like(rays[...,0], dtype=torch.long)
        # ray_idx_map[rays[...,0].long()] = torch.arange(rays.shape[0], device=ray_idx_map.device)
        import matplotlib.pyplot as plt
        loc_depth = ((xyzs - ray_o) / dirs)[...,1].detach()
        blend_weight = weights.detach()
        sdfs = sdfs.detach()
        size = self.opt.debug_patch_size
        step = self.opt.debug_patch_step

        for x in range(0, size, step):
            for y in range(0, size, step):
                idx = x * size + y
                fig, ax1 = plt.subplots(figsize=(10, 5))
                # add legend and axis labels
                ax1.plot(loc_depth[idx, :].cpu(), blend_weight[idx, :].cpu(), color='blue', label='blend weight')
                ax1.scatter(loc_depth[idx, :].cpu(), blend_weight[idx, :].cpu(), color='blue', label='blend weight')
                ax2 = ax1.twinx()
                ax2.plot(loc_depth[idx, :].cpu(), sdfs[idx, :].cpu(), color='g', label='sdf')
                ax2.scatter(loc_depth[idx, :].cpu(), sdfs[idx, :].cpu(), color='g', label='sdf')
                ax1.set_xlabel('sample depth')
                ax1.set_ylabel('blend weight')
                ax2.set_ylabel('sdf')
                # ax1.legend(loc='upper left')
                # if xlim is not None: plt.xlim(right=xlim)
                plt.savefig(f'{debug_path}/curve_{x:01d}_{y:01d}.png')

        normal_image = F.normalize(normal_image, dim=-1) * 0.5 + 0.5
        plt.imsave(f'{debug_path}/normal.png', normal_image.reshape(size, size, -1).detach().cpu().numpy()[...,[2,0,1]])
        plt.imsave(f'{debug_path}/image.png', image.reshape(size, size, -1).detach().cpu().numpy().clip(0,1.))
        print('Beta: ', self.sdf_density.get_beta())
        # import IPython; IPython.embed()
        # exit()

    return results

def unwrap_env_sphere(trainer, device, dt_gamma=0, bg_color=None, num_step=1, step_size=0.002, use_specular_color=True, env_net_index=None, material=None, r_images=None, **kwargs):
    from .lighting_util import gen_light_xyz
    # test only (SDF & ENV mlp loaded from ckpt)

    # rays_o, rays_d: [B, N, 3], assumes B == 1
    # return: image: [B, N, 3], depth: [B, N]
    self = trainer.model

    from nerf.provider import get_pose

    poses = get_pose(device, 0.0, 0.0, 4.0)
    rays_o = poses[..., :3, 3]
    sphere_center = torch.tensor([0.0, 0.0, 0.0]).to(device)
    directions = sphere_center - rays_o
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions # @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    prefix = rays_o.shape[:-1]
    rays_o = rays_o.contiguous().view(-1, 3)
    # TODO: rays_d should be camera to center of sphere
    rays_d = rays_d.contiguous().view(-1, 3)

    device = rays_o.device

    env_h, env_w = 512, 1024
    lxyz, lareas = gen_light_xyz(env_h, env_w, 1.0)

    lxyz = lxyz.reshape(-1, lxyz.shape[-1])
    lxyz = lxyz[..., [1, 2, 0]]
    w_r = lxyz  / np.linalg.norm(lxyz, axis=-1, keepdims=True)
    w_r = torch.Tensor(w_r).to(device)
    n_pixels = w_r.shape[0]

    # get near and far
    nears, fars, mask = get_sphere_intersections(rays_o, rays_d, self.opt.env_sph_radius) # [N, 1], [N, 1], [N, 1]

    nears_valid = nears[mask] # [M, 1]

    # z_radius = step_size * (num_step-1) / 2
    # z_vals = torch.linspace(-z_radius, z_radius, num_step, device=device)[None, :] # [1, S]
    # z_vals = z_vals + nears_valid # [M, S]
    z_vals = nears_valid
    
    dirs = rays_d[mask, None, :] # [M, 1, 3]
    xyzs = rays_o[mask, None, :] + dirs * z_vals[:, :, None] # [M, S, 3]

    dirs = dirs.reshape(-1, dirs.shape[-1])
    xyzs = xyzs.reshape(-1, xyzs.shape[-1])

    use_normals = True
    
    if use_normals:
        xyzs.requires_grad = True
        
        # sdfs, sigmas, rgbs = self(xyzs, dirs)

    sdfs, sigmas, geo_feats, normals, _ = self.forward_sigma(xyzs, use_sdf_sigma_grad=use_normals, material=material) 
    geo_feats = geo_feats.repeat((n_pixels, 1))
    normals = -dirs.repeat((n_pixels, 1))

    # roughness = 0.0 # min roughness
    roughness = getattr(self, 'roughness', 0.0)
    # import IPython; IPython.embed()

    with torch.set_grad_enabled(False):
        # sigmas = self.density_scale * sigmas
        
        normals_enc, _, n_dot_w_o, n_env_enc = self.get_color_mlp_extra_params(normals, dirs, roughness)
        w_r_enc = self.encoder_refdir(w_r, roughness=roughness)

        r_images = None

        rgbs = self.forward_color(geo_feats, dirs, normals_enc, w_r_enc, n_dot_w_o, use_specular_color, env_net_index, n_env_enc=n_env_enc, r_images=r_images, roughness=roughness)

    img = rgbs.reshape(env_h, env_w, 3)
    if self.opt.color_space == 'linear':
        img = linear_to_srgb(img)
    img = img.detach().cpu().numpy().clip(0, 1)
    img = (img * 255).astype(np.uint8)
    import imageio
    import os
    swap_env_path = trainer.opt.swap_env_path
    env_net_basename = os.path.basename(swap_env_path)
    env_net_basename = os.path.splitext(env_net_basename)[0]
    output_name = f"unwrap_env{ '_' + str(env_net_index) if env_net_index is not None else ''}_r{self.opt.unwrap_roughness}.png"
    print(f"unwrapped to {self.opt.workspace}/{output_name}")
    imageio.imwrite(f"{self.opt.workspace}/{output_name}", img)