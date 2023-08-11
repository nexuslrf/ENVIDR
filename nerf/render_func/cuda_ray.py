import torch 

from typing import Optional
import torch
import torch.nn.functional as F

import torch.nn as nn
from torch import Tensor

import raymarching
from .utils import curve_plot

SQRT3 = 3**0.5

def run_cuda(model, rays_o, rays_d, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4,
        get_normal_image=False, use_specular_color=True, early_stop_steps=-1, ray_depth=None, 
        main_pass=True, r_images=None, geometry_only=False, grad_ray=False, bg_sphere=True, env_rot_radian=None,
        **kwargs):
    # rays_o, rays_d: [B, N, 3], assumes B == 1
    # return: image: [B, N, 3], depth: [B, N]
    self = model

    prefix = rays_o.shape[:-1]
    rays_o = rays_o.contiguous().view(-1, 3)
    rays_d = rays_d.contiguous().view(-1, 3)

    debug = self.opt.debug
    use_relsdf_loss = self.opt.relsdf_loss or self.opt.dist_bound
    use_orientation_loss = self.opt.orientation_loss
    use_backsdf_loss = self.opt.backsdf_loss
    use_eikonal_loss = self.opt.eikonal_loss
    use_normal_with_mlp = self.use_normal_with_mlp
    use_neus = self.opt.use_neus_sdf
    use_sdf_sigma_grad = debug or use_relsdf_loss or use_backsdf_loss or use_eikonal_loss or use_orientation_loss \
        or use_normal_with_mlp or self.use_n_dot_viewdir or self.use_reflected_dir or use_neus

    if debug and not self.training:
        force_all_rays = True

    N = rays_o.shape[0] # N = B * N, in fact
    device = rays_o.device

    # pre-calculate near far
    if ray_depth is not None:
        dt = 2 * SQRT3 / max_steps
        valid_ray = (ray_depth > 0)
        nears = (ray_depth - 4*dt) * valid_ray
        fars = (ray_depth + 4*dt) * valid_ray
        # early_stop_steps = 10
    else:
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)

    results = {}

    # mix background color
    if self.bg_radius > 0 and bg_sphere:
        # use the bg model to calculate bg_color
        sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
        bg_color = self.background(sph, rays_d) # [N, 3]
        results['sphere_bg'] = bg_color
    elif bg_color is None:
        bg_color = 1    
    
    if self.training or debug:
        # setup counter
        if not force_all_rays:
            counter = self.step_counter[self.local_step % 16]
            counter.zero_() # set to 0
            self.local_step = self.local_step + 1
        else:
            counter = None
        
        # counter = self.step_counter[self.local_step % 16]
        # counter.zero_() # set to 0
        # self.local_step = self.local_step + 1

        with torch.no_grad(): # march_rays kernel does not support backward
            perturb = (not self.opt.stratified_sampling) and perturb
            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps, early_stop_steps)
            if self.opt.stratified_sampling and self.training:
                dt = 2 * SQRT3 / max_steps # only work for non-cascade occ.
                strat_scale = 0.5
                strat_noise = torch.rand_like(deltas[...,:1]) * 2 - 1 # [-1, 1]
                strat_noise = strat_noise * strat_scale * dt
                strat_noise_roll = strat_noise.roll(1, dims=-2)
                strat_detlas = strat_noise_roll - strat_noise
                deltas = deltas + strat_detlas
                # import IPython; IPython.embed()
                xyzs = xyzs + strat_detlas * dirs
                
            
        if use_sdf_sigma_grad:
            xyzs.requires_grad = True
            # xyzs = xyzs - rays_o.detach() + rays_o
        scatter_idx = None
        if r_images is not None:
            scatter_idx = raymarching.get_scatter_idx(rays, rays.new_zeros(xyzs.shape[0])).long()
            r_images = r_images[0, scatter_idx] if r_images is not None else None

        if grad_ray:
            if scatter_idx is None:
                scatter_idx = raymarching.get_scatter_idx(rays, rays.new_zeros(xyzs.shape[0])).long()
            dirs = rays_d[scatter_idx]
            rays_o_scatter = rays_o[scatter_idx]
            xyzs = xyzs - self.opt.grad_rays_scale * rays_o_scatter.detach() + self.opt.grad_rays_scale * rays_o_scatter
        #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())
        
        # sdfs, sigmas, rgbs = self(xyzs, dirs)

        sdfs, sigmas, geo_feats, normals, eikonal_sdf_gradients = self.forward_sigma(xyzs, use_sdf_sigma_grad=use_sdf_sigma_grad, dirs=dirs, dists=deltas[...,0])

        # density_outputs = self.density(xyzs) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
        # sigmas = density_outputs['sigma']
        # rgbs = self.color(xyzs, dirs, **density_outputs)
        sigmas = self.density_scale * sigmas
        roughness = getattr(self, 'roughness', self.opt.default_roughness)
        

        if geometry_only:
            with torch.set_grad_enabled(not self.opt.detach_normal):
                weights_sum, depth, normal_image, weights = raymarching.composite_rays_train(sigmas, normals, deltas, rays, T_thresh, False, use_neus)
            depth = (depth + nears) * (depth != 0)
            depth = depth.view(*prefix)
            results['normal_image'] = F.normalize(normal_image, dim=-1).view(*prefix, 3)
            image = None
        else:
            normals_enc, w_r_enc, n_dot_w_o, n_env_enc = self.get_color_mlp_extra_params(normals, dirs, roughness)

            rgbs = self.forward_color(geo_feats, dirs, normals_enc, w_r_enc, n_dot_w_o, use_specular_color, n_env_enc=n_env_enc, r_images=r_images, roughness=roughness)

            #print(f'valid RGB query ratio: {mask.sum().item() / mask.shape[0]} (total = {mask.sum().item()})')

            # special case for CCNeRF's residual learning
            if len(sigmas.shape) == 2:
                K = sigmas.shape[0]
                depths = []
                images = []
                for k in range(K):
                    weights_sum, depth, image = raymarching.composite_rays_train(sigmas[k], rgbs[k], deltas, rays, T_thresh, False, use_neus)
                    image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                    depth = torch.clamp(depth - nears, min=0) / (fars - nears)
                    images.append(image.view(*prefix, 3))
                    depths.append(depth.view(*prefix))
            
                depth = torch.stack(depths, axis=0) # [K, B, N]
                image = torch.stack(images, axis=0) # [K, B, N, 3]

            else:
                # weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, T_thresh)
                ret_weights = False 
                if debug or use_backsdf_loss or use_relsdf_loss or use_orientation_loss or self.opt.weighted_eikonal:
                    ret_weights = True
                # TODO: loss (10) (11)
                weights_sum, depth, image, weights = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, T_thresh, ret_weights, use_neus)
                image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                # depth = torch.clamp(depth, min=0) / (fars - nears)
                depth = (depth + nears) * (depth != 0)
                image = image.view(*prefix, 3)
                depth = depth.view(*prefix)
        
        results['weights_sum'] = weights_sum.view(*prefix)
        results['sigmas'] = sigmas
        results['sdfs'] = sdfs
        if self.opt.cauchy_roughness_weighted:
            results['roughness'] = roughness
        if use_eikonal_loss:
            if self.opt.weighted_eikonal:
                results['sdf_gradients'] = eikonal_sdf_gradients * weights.detach()[...,None]
            else:
                results['sdf_gradients'] = eikonal_sdf_gradients

        # TODO 
        if main_pass and (use_relsdf_loss or use_backsdf_loss or use_orientation_loss):
            point_mask = torch.ones_like(sigmas, dtype=torch.bool)
            M = sigmas.shape[0]
            ray_valid = (rays[:,2] > 0) * (rays[:,1] + rays[:,2] < M)
            # ray_valid_idx = ray_valid.nonzero(as_tuple=False).squeeze(-1)
            ray_valid_start_pnt_idx = rays[ray_valid, 1]
            ray_valid_end_pnt_idx = ray_valid_start_pnt_idx + rays[ray_valid, 2] - 1
            total_valid_pnts = rays[ray_valid, 2].sum()
            # mark the last sampled point of each ray as invalid. E.g., start_pnt_idx=0 -1 => -1 mark the last ray's last point as invalid
            point_mask[ray_valid_end_pnt_idx.long()] = False
            # deltas[i, 0] is the step size from the i-th point, 
            # deltas[i, 1] is the distance from the i-1 -th point to the i-th point
            # apparently, the first point of a ray do not have proper distance. we will shift the deltas to make it the distance from the i-th point to the i+1-th point
            deltas_shifted = torch.roll(deltas, -1, 0)
            delta_mask = (deltas_shifted[:, 0] > 0) * (deltas_shifted[:, 1] > 0)
            cont_mask = deltas_shifted[:, 1] < 1.2 * deltas_shifted[:, 0]
            point_mask = point_mask * delta_mask*cont_mask
            
            if self.obj_aabb is not None:
                bound_mask = (xyzs >= self.obj_aabb[:3]).all(-1) * (xyzs <= self.obj_aabb[3:]).all(-1)
                point_mask = point_mask * bound_mask
            # now compute the relsdf
            if not self.use_sdf:
                sdfs = -sigmas
            relsdf = torch.roll(sdfs, -1, dims=0) - sdfs # sdfs is decreasing along the ray, so relsdf should be negative
            # next is to compute the cos, make sure xyz_grads is normalized
            cos = (dirs * normals).sum(-1) # this is cos of \pi/2 - \theta
            # now we can compute the approximated relsdf
            # import IPython; IPython.embed()

            est_relsdf = deltas_shifted[:, 1] * cos.detach() # in most cases, this should be negative number
            results['relsdf'] = relsdf[point_mask]
            results['est_relsdf'] = est_relsdf[point_mask]
            results['cos'] = cos[point_mask]
            # TODO: do I need to return the weights here? (And do I need to shift the weights? No)
            # if use_backsdf_loss:
            results['sdf_weights'] = weights[point_mask]
            results['sdf_dist'] = deltas_shifted[point_mask, 1]
            results['sdfs'] = sdfs[point_mask]

            # results['weights'] = weights
        if debug and not self.training and main_pass:
            import os
            workspace = self.opt.workspace
            epoch = kwargs.get('epoch', 0)
            debug_path = os.path.join(workspace, 'debug', f'ep{epoch}')
            os.makedirs(debug_path, exist_ok=True)
            # ray_o = rays_o[0].view(-1, 3)
            ray_idx_map = torch.ones_like(rays[...,0], dtype=torch.long)
            ray_idx_map[rays[...,0].long()] = torch.arange(rays.shape[0], device=ray_idx_map.device)
            import matplotlib.pyplot as plt
            # loc_depth = ((xyzs - ray_o) / dirs)[...,1].detach()
            blend_weight = weights.detach()
            sdfs = sdfs.detach()
            size = self.opt.debug_patch_size
            step = self.opt.debug_patch_step
            for x in range(0, size, step):
                for y in range(0, size, step):
                    curve_plot(x, y, size, ray_idx_map, rays, rays_o, rays_d, xyzs, blend_weight, sdfs, debug_path)
            weights_sum, depth, normal_image, weights = raymarching.composite_rays_train(sigmas, normals.detach(), deltas, rays, T_thresh, ret_weights, use_neus)
            normal_image = F.normalize(normal_image, dim=-1) * 0.5 + 0.5
            plt.imsave(f'{debug_path}/normal.png', normal_image.reshape(size, size, -1).detach().cpu().numpy()[...,[2,0,1]])
            plt.imsave(f'{debug_path}/image.png', image.reshape(size, size, -1).detach().cpu().numpy().clip(0,1.))
            # import IPython; IPython.embed()
            # exit()
    else:
        
        # allocate outputs 
        # if use autocast, must init as half so it won't be autocasted and lose reference.
        #dtype = torch.half if torch.is_autocast_enabled() else torch.float32
        # output should always be float32! only network inference uses half.
        dtype = torch.float32
        
        weights_sum = torch.zeros(N, dtype=dtype, device=device)
        depth = torch.zeros(N, dtype=dtype, device=device)
        image = torch.zeros(N, 3, dtype=dtype, device=device)
        
        n_alive = N
        rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
        rays_t = nears.clone() # [N]

        # normal image
        if get_normal_image:
            normal_weights_sum = torch.zeros(N, dtype=dtype, device=device)
            normal_rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            normal_rays_t = nears.clone() # [N]
            normal_image = torch.zeros(N, 3, dtype=dtype, device=device) # [N, 3]
            normal_depth = torch.zeros(N, dtype=dtype, device=device)

        if self.opt.use_diffuse:
            if 'diffuse' in self.opt.visual_items:
                diffuse_rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
                diffuse_image = torch.zeros(N, 3, dtype=dtype, device=device) # [N, 3]
                diffuse_weights_sum = torch.zeros(N, dtype=dtype, device=device)
                diffuse_rays_t = nears.clone() # [N]
                diffuse_depth = torch.zeros(N, dtype=dtype, device=device)
            if 'specular' in self.opt.visual_items:
                specular_rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device)
                specular_image = torch.zeros(N, 3, dtype=dtype, device=device)
                specular_weights_sum = torch.zeros(N, dtype=dtype, device=device)
                specular_rays_t = nears.clone() # [N]
                specular_depth = torch.zeros(N, dtype=dtype, device=device)

        step = 0
        while step < max_steps:

            # count alive rays 
            n_alive = rays_alive.shape[0]
            
            # exit loop
            if n_alive <= 0:
                break

            # decide compact_steps
            n_step = max(min(N // n_alive, 8), 1)
            align = 128
            # TODO:
            _r_images = None
            if r_images is not None:
                _r_images = r_images[0, rays_alive.long()][:, None, :].expand(-1, n_step, -1).reshape(-1, r_images.shape[-1])
                align = -1

            xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, align, perturb if step == 0 else False, dt_gamma, max_steps)

            if get_normal_image:
                xyzs.requires_grad = True # for normal calculation
                use_sdf_sigma_grad = True
            
            # sdfs, sigmas, rgbs = self(xyzs, dirs)

            sdfs, sigmas, geo_feats, sigma_grads, _ = self.forward_sigma(xyzs, use_sdf_sigma_grad=use_sdf_sigma_grad, dirs=dirs, dists=deltas[...,0])
            roughness = getattr(self, 'roughness', self.opt.default_roughness)
            sigmas = self.density_scale * sigmas
            # TODO: add geometry_only branch
            if geometry_only:
                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, sigma_grads, deltas, weights_sum, depth, normal_image, T_thresh, use_neus)
            else:
                normals_enc, w_r_enc, n_dot_w_o, n_env_enc = self.get_color_mlp_extra_params(sigma_grads, dirs, roughness, env_rot_radian)
                rgbs = self.forward_color(geo_feats, dirs, normals_enc, w_r_enc, n_dot_w_o, use_specular_color, n_env_enc=n_env_enc, r_images=_r_images, roughness=roughness)

                # density_outputs = self.density(xyzs) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
                # sigmas = density_outputs['sigma']
                # rgbs = self.color(xyzs, dirs, **density_outputs)
                # sigmas = self.density_scale * sigmas

                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh, use_neus)

                if self.opt.use_diffuse:
                    # print("composite diffuse")
                    if 'diffuse' in self.opt.visual_items:
                        c_diffuse = self.c_diffuse
                        raymarching.composite_rays(n_alive, n_step, diffuse_rays_alive, diffuse_rays_t, sigmas, c_diffuse, deltas, diffuse_weights_sum, diffuse_depth, diffuse_image, T_thresh, use_neus)
                        results["diffuse_image"] = diffuse_image
                        diffuse_rays_alive = diffuse_rays_alive[diffuse_rays_alive >= 0]
                    if 'specular' in self.opt.visual_items:
                        c_specular = self.c_specular
                        accum_deltas = True
                        if isinstance(roughness, torch.Tensor):
                            deltas[...,1:] = roughness
                            accum_deltas = False
                        raymarching.composite_rays(n_alive, n_step, specular_rays_alive, specular_rays_t, sigmas, c_specular, deltas, specular_weights_sum, specular_depth, specular_image, T_thresh, use_neus, accum_deltas)
                        results["specular_image"] = specular_image
                        results["roughness_image"] = specular_depth[...,None]
                        specular_rays_alive = specular_rays_alive[specular_rays_alive >= 0]
                        
                # raymarching for normal image
                if get_normal_image:
                    raymarching.composite_rays(n_alive, n_step, normal_rays_alive , normal_rays_t, sigmas, sigma_grads, deltas, normal_weights_sum, normal_depth, normal_image, T_thresh, use_neus)
                    # print("normal_image has nan={}".format(torch.isnan(normal_image).any()))
                    normal_rays_alive = normal_rays_alive[normal_rays_alive >= 0]

            #print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')
            rays_alive = rays_alive[rays_alive >= 0]
            step += n_step

        if image is not None:
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            image = image.view(*prefix, 3)
        # depth = torch.clamp(depth - nears, min=0) / (fars - nears)
        depth = depth.view(*prefix)
        results['weights_sum'] = weights_sum.view(*prefix)

        # normal image
        if get_normal_image:
            normal_image = F.normalize(normal_image, dim=-1, eps=1e-10)
            normal_image = normal_image.view(*prefix, 3)
            results['normal_image'] = normal_image
    
    results['depth'] = depth
    results['image'] = image

    return results