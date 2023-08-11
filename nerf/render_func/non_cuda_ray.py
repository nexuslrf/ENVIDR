import torch 

from typing import Optional
import torch
import torch.nn.functional as F

import torch.nn as nn
from torch import Tensor

import raymarching
from .utils import sample_pdf

def run(model, rays_o, rays_d, num_steps=128, upsample_steps=128, bg_color=None, perturb=False, get_normal_image=False, use_specular_color=True, **kwargs):
    # rays_o, rays_d: [B, N, 3], assumes B == 1
    # bg_color: [3] in range [0, 1]
    # return: image: [B, N, 3], depth: [B, N]
    self = model

    prefix = rays_o.shape[:-1]
    rays_o = rays_o.contiguous().view(-1, 3)
    rays_d = rays_d.contiguous().view(-1, 3)

    debug = self.opt.debug
    # TODO: backsdf for non-cuda-ray mode is not implemented yet.
    use_backsdf_loss = self.opt.backsdf_loss
    use_eikonal_loss = self.opt.eikonal_loss
    use_normal_with_mlp = self.use_normal_with_mlp
    use_neus_density = True
    use_sdf_sigma_grad = debug or use_backsdf_loss or use_eikonal_loss or use_normal_with_mlp or self.use_n_dot_viewdir or self.use_reflected_dir or use_neus_density

    N = rays_o.shape[0] # N = B * N, in fact
    device = rays_o.device

    # choose aabb
    aabb = self.aabb_train if self.training else self.aabb_infer

    # sample steps
    nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
    nears.unsqueeze_(-1)
    fars.unsqueeze_(-1)

    #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

    z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
    z_vals = z_vals.expand((N, num_steps)) # [N, T]
    z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

    # perturb z_vals
    sample_dist = (fars - nears) / num_steps
    if perturb:
        z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
        #z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

    # generate xyzs
    xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
    xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.
    if get_normal_image or use_sdf_sigma_grad:
        xyzs.requires_grad_(True)

    #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

    # query SDF and RGB
    #         use_sdf_sigma_grad = kwargs.get("use_sdf_sigma_grad", False)
    # dirs = kwargs.get("dirs", None)
    # deltas = kwargs.get("deltas", None)
    flattened_xyzs = xyzs.reshape(-1, 3)
    flattened_dirs = rays_d.reshape(-1, 3).repeat(num_steps, 1)
    flattened_dists = z_vals.reshape(-1, )
    density_outputs = self.density(flattened_xyzs, use_sdf_sigma_grad=use_sdf_sigma_grad, dirs=flattened_dirs, dists=flattened_dists)

    #sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
    for k, v in density_outputs.items():
        if density_outputs[k] is not None:
            density_outputs[k] = v.view(N, num_steps, -1)

    # upsample z_vals (nerf-like)
    if upsample_steps > 0:
        with torch.no_grad():

            deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
            deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

            alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1)) # [N, T]
            alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+1]
            weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T]

            # sample new z_vals
            z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [N, T-1]
            new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps, det=not self.training).detach() # [N, t]

            new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
            new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:]) # a manual clip.

        if get_normal_image or use_sdf_sigma_grad:
            new_xyzs.requires_grad_(True)
        # only forward new points to save computation
        flattened_new_xyzs = new_xyzs.reshape(-1, 3)
        flattened_new_dirs = rays_d.reshape(-1, 3).repeat(upsample_steps, 1)
        flattened_new_dists = new_z_vals.reshape(-1, )
        new_density_outputs = self.density(flattened_new_xyzs, use_sdf_sigma_grad=True, dirs=flattened_new_dirs, dists=flattened_new_dists)

        #new_sigmas = new_density_outputs['sigma'].view(N, upsample_steps) # [N, t]
        for k, v in new_density_outputs.items():
            if new_density_outputs[k] is not None:
                new_density_outputs[k] = v.view(N, upsample_steps, -1)

        # re-order
        z_vals = torch.cat([z_vals, new_z_vals], dim=1) # [N, T+t]
        z_vals, z_index = torch.sort(z_vals, dim=1)

        xyzs = torch.cat([xyzs, new_xyzs], dim=1) # [N, T+t, 3]
        xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

        for k in density_outputs:
            if density_outputs[k] is not None:
                assert new_density_outputs[k] is not None, f"new_density_outputs[{k}] is None"
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

    with torch.set_grad_enabled(self.training):
        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1)) # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            if density_outputs[k] is not None:
                density_outputs[k] = v.view(-1, v.shape[-1])

        mask = weights > 1e-4 # hard coded
        rgbs = self.color(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), mask=mask.reshape(-1), **density_outputs)
        rgbs = rgbs.view(N, -1, 3) # [N, T+t, 3]

        #print(xyzs.shape, 'valid_rgb:', mask.sum().item())

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [N]

    # TODO normal
    if get_normal_image:
        sigma_grads = density_outputs['normal'].reshape(*weights.shape, 3)
        normal_image = torch.sum(weights.unsqueeze(-1) * sigma_grads, dim=-2)
        normal_image = F.normalize(normal_image, dim=-1, eps=1e-10)
    else:
        normal_image = None

    # calculate depth 
    ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
    depth = torch.sum(weights * ori_z_vals, dim=-1)

    # calculate color
    image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]

    # mix background color
    if self.bg_radius > 0:
        # use the bg model to calculate bg_color
        sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
        bg_color = self.background(sph, rays_d.reshape(-1, 3)) # [N, 3]
    elif bg_color is None:
        bg_color = 1
        
    image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

    image = image.view(*prefix, 3)
    depth = depth.view(*prefix)

    # tmp: reg loss in mip-nerf 360
    # z_vals_shifted = torch.cat([z_vals[..., 1:], sample_dist * torch.ones_like(z_vals[..., :1])], dim=-1)
    # mid_zs = (z_vals + z_vals_shifted) / 2 # [N, T]
    # loss_dist = (torch.abs(mid_zs.unsqueeze(1) - mid_zs.unsqueeze(2)) * (weights.unsqueeze(1) * weights.unsqueeze(2))).sum() + 1/3 * ((z_vals_shifted - z_vals_shifted) * (weights ** 2)).sum()

    ret = {
        'depth': depth,
        'image': image,
        'weights_sum': weights_sum,
        'normal_image': normal_image
    }
    if use_eikonal_loss and self.training:
        ret['sdf_gradients'] = density_outputs['sdf_gradients']
    return ret