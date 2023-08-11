import torch 

from typing import Optional
import torch
import torch.nn.functional as F

import torch.nn as nn
from torch import Tensor

import raymarching


# @torch.jit.script
def aabb_intersect_bbox(ray_o: torch.Tensor, ray_d: torch.Tensor, 
            bbox: torch.Tensor, near: float, far: float):

    tbot = (bbox[:3] - ray_o) / ray_d
    ttop = (bbox[3:] - ray_o) / ray_d
    tmin = torch.where(tbot < ttop, tbot, ttop)
    tmax = torch.where(tbot > ttop, tbot, ttop)
    largest_tmin, _ = tmin.max(dim=1)
    smallest_tmax, _ = tmax.min(dim=1)
    t_near = largest_tmin.clamp_min(near)
    t_far = smallest_tmax.clamp_max(far)
    return t_near, t_far

def get_error_bound(
        beta: Tensor, sdf: Tensor, dists: Tensor, d_star: Tensor
    ) -> Tensor:
    density = volsdf_density_fn(sdf, beta=beta)
    shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), dists * density[:, :-1]], dim=-1)
    integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
    error_per_section = torch.exp(-d_star / beta) * (dists ** 2.) / (4 * beta ** 2)
    error_integral = torch.cumsum(error_per_section, dim=-1)
    bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * torch.exp(-integral_estimation[:, :-1])

    return bound_opacity.max(-1)[0]

def volsdf_density_fn(sdf: Tensor, beta: Tensor):
    alpha = 1 / beta
    return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

# @torch.jit.script
# @torch.no_grad()
def cdf_sample(
    N_samples: int,
    z_vals: Tensor, weights: Tensor,
    det: bool=True, mid_bins: bool=True,
    eps: float=1e-5, include_init_z_vals: bool=True
    ):
    '''
    cdf_sample relies on (i) coarse_sample's results (ii) output of coarse MLP
    In this function, each ray will have the same number of sampled points, 
    there's  voxel_cdf_sample function, that sample variable points for different rays.
    TODO@chensjtu we also plan to write a CUDA version of normal cdf sample, which can avoid using sort.
    Args:
        rays_o: Tensor, the orgin of rays. [N_rays, 3]
        rays_d: Tensor, the direction of rays. [N_rays, 3]
        z_vals: Tensor, samples positional parameter in coarse sample. [N_rays|1, N_samples]
        weights: Tensor, processed weights from MLP and vol rendering. [N_rays, N_samples]
    '''
    device = weights.device
    N_rays = weights.shape[0]
    N_base_samples = z_vals.shape[1]
    if mid_bins:
        bins = .5 * (z_vals[...,1:] + z_vals[...,:-1]) 
        weights_ = weights[..., 1:-1] + eps # prevent nans
    else:
        bins = z_vals
        weights_ = weights[..., :-1] + eps # prevent nans
    
    # Get pdf & cdf
    pdf = weights_ / torch.sum(weights_, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(N_rays, N_samples)
    else:
        u = torch.rand(N_rays, N_samples, device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = (inds-1).clamp_min(0)
    above = inds.clamp_max(cdf.shape[-1]-1)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom[denom < 1e-5] = 1
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
    if include_init_z_vals:
        z_samples, _ = torch.sort(torch.cat([z_vals.expand(N_rays, N_base_samples), samples], -1), -1)
    else:
        z_samples, _ = torch.sort(samples, -1)
    
    return z_samples

# @torch.jit.script # this loop body is a good candidate for JIT.
def error_bound_sampling_update(
        z_vals: Tensor, sdf: Tensor, beta0: Tensor, beta: Tensor, eps: float, beta_iters: int,
        last_iter: bool, add_tiny: float, N_samples: int, N_samples_eval: int, det: bool
    ):
    device = z_vals.device
    # Calculating the bound d* (Theorem 1)
    d = sdf # [N, S]
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
    first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
    second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
    d_star = torch.zeros(z_vals.shape[0], z_vals.shape[1] - 1, device=device)
    d_star[first_cond] = b[first_cond]
    d_star[second_cond] = c[second_cond]
    s = (a + b + c) / 2.0
    area_before_sqrt = s * (s - a) * (s - b) * (s - c)
    mask = ~first_cond & ~second_cond & (b + c - a > 0)
    d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
    d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star  # Fixing the sign

    # Updating beta using line search
    curr_error = get_error_bound(beta0, sdf, dists, d_star)
    beta[curr_error <= eps] = beta0
    beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
    for j in range(beta_iters):
        beta_mid = (beta_min + beta_max) / 2.
        curr_error = get_error_bound(beta_mid.unsqueeze(-1), sdf, dists, d_star)
        beta_max[curr_error <= eps] = beta_mid[curr_error <= eps]
        beta_min[curr_error > eps] = beta_mid[curr_error > eps]
    beta = beta_max

    # Upsample more points
    density = volsdf_density_fn(sdf, beta=beta.unsqueeze(-1))

    dists = torch.cat([dists, 1e10*torch.ones(dists.shape[0],1, device=device)], -1)
    free_energy = dists * density
    shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1, device=device), free_energy[:, :-1]], dim=-1)
    alpha = 1 - torch.exp(-free_energy)
    transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))

    #  Check if we are done and this is the last sampling
    not_converge = beta.max() > beta0
    add_more = not_converge and not last_iter

    if add_more:
        ''' Sample more points proportional to the current error bound'''
        N = N_samples_eval
        error_per_section = torch.exp(-d_star / beta.unsqueeze(-1)) * (dists[:,:-1] ** 2.) / (4 * beta.unsqueeze(-1) ** 2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral),max=1.e6) - 1.0) * transmittance[:,:-1]
        weights = torch.cat([bound_opacity, torch.ones_like(bound_opacity[:,:1])], -1)
        cdf_eps = add_tiny
    else:
        ''' Sample the final sample set to be used in the volume rendering integral '''
        N = N_samples
        weights = alpha * transmittance  # probability of the ray hits something here
        cdf_eps = 1e-5
    
    cdf_det = add_more or det
    samples = cdf_sample(N, z_vals, weights, cdf_det, mid_bins=False, eps=cdf_eps, include_init_z_vals=False)
    # Adding samples if we not converged
    if add_more:
        z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)
    else:
        samples_idx = None
    return samples, z_vals, samples_idx, beta, not_converge

def run_volsdf(model, rays_o, rays_d, num_steps=128, upsample_steps=64, bg_color=None, perturb=False, get_normal_image=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        self = model

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        debug = self.opt.debug
        # TODO: and backsdf for non-cuda-ray mode is not implemented yet.
        use_backsdf_loss = self.opt.backsdf_loss
        use_eikonal_loss = self.opt.eikonal_loss
        use_normal_with_mlp = self.use_normal_with_mlp
        use_sdf_sigma_grad = debug or use_backsdf_loss or use_eikonal_loss or use_normal_with_mlp or self.use_n_dot_viewdir or self.use_reflected_dir

        N_ori = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)
        valid_mask = (nears < fars).squeeze(-1)

        if valid_mask.any():
            rays_o = rays_o[valid_mask]
            rays_d = rays_d[valid_mask]
            nears = nears[valid_mask]
            fars = fars[valid_mask]
            N = rays_o.shape[0]

            #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')
            z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
            z_vals = z_vals.expand((N, num_steps)) # [N, T]
            z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

            # perturb z_vals
            sample_dist = (fars - nears) / num_steps
            if perturb:
                z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
                # z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

            # run volsdf sampling
            

            samples, samples_idx = z_vals, None
            sdf = None
            beta0 = self.sdf_density.get_beta().detach()

            # hyperparams:
            eps = 0.1
            max_total_iters = 5
            beta_iters = 10
            add_tiny = 1e-6
            N_samples_extra = 32

            # Get maximum beta from the upper bound (Lemma 2)
            dists = z_vals[:, 1:] - z_vals[:, :-1] # [N, T-1]
            bound = (1.0 / (4.0 * torch.log(torch.tensor(eps + 1.0)))) * (dists ** 2.).sum(-1)
            beta = torch.sqrt(bound)

            total_iters, not_converge = 0, True

            while not_converge and total_iters < max_total_iters:
                points = rays_o.unsqueeze(1) + samples.unsqueeze(2) * rays_d.unsqueeze(1)
                points = torch.min(torch.max(points, aabb[:3]), aabb[3:]) # a manual clip.
                points_flat = points.reshape(-1, 3)

                # Calculating the SDF only for the new sampled points
                with torch.no_grad():
                    samples_sdf, sigmas, _ = self.forward_sigma(points_flat)
                if sdf is not None and samples_idx is not None:
                    sdf_merge = torch.cat([sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                                        samples_sdf.reshape(-1, samples.shape[1])], -1)
                    sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
                else:
                    sdf = samples_sdf

                sdf = sdf.reshape_as(z_vals) # [N_rays, N_samples]
                total_iters += 1
                last_iter = (total_iters == max_total_iters)
                samples, z_vals, samples_idx, beta, not_converge = error_bound_sampling_update(
                    z_vals, sdf, beta0, beta, eps, beta_iters, last_iter,
                    add_tiny, upsample_steps, num_steps, (not self.training)
                )
            
            z_samples = samples

            if N_samples_extra > 0:
                if self.training:
                    sampling_idx = torch.randperm(z_vals.shape[1])[:N_samples_extra]
                else:
                    sampling_idx = torch.linspace(0, z_vals.shape[1]-1, N_samples_extra).long()
                z_vals_extra = torch.cat([nears, fars, z_vals[:,sampling_idx]], -1)
            else:
                z_vals_extra = torch.cat([nears, fars], -1)

            z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)

            xyzs = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,None] # [N_rays, N_samples, 3]
            xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.
            # end of sampling
            # add some of the near surface points
            xyzs_eik_near = None
            if use_eikonal_loss and self.training:
                idx = torch.randint(z_samples.shape[-1], (z_samples.shape[0],), device=device)
                z_samples_eik = torch.gather(z_samples, 1, idx.unsqueeze(-1))
                xyzs_eik_near = rays_o[...,None,:] + rays_d[...,None,:] * z_samples_eik[...,None]
                
            # query SDF and RGB
            if get_normal_image or use_sdf_sigma_grad:
                xyzs.requires_grad_(True)

            density_outputs = self.density(xyzs.reshape(-1, 3))

            if get_normal_image or use_sdf_sigma_grad:
                if self.use_sdf:
                    normals = self.get_grad(density_outputs['sdf'], xyzs) # .detach()
                else:
                    normals = self.get_grad(density_outputs['sigma'], xyzs) # .detach()
                
                normals = normals.detach() if self.opt.detach_normal else normals
                normals = F.normalize(normals, dim=-1, eps=1e-10)
                density_outputs['normal'] = normals

            #sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
            num_z_steps = xyzs.shape[1]
            for k, v in density_outputs.items():
                # import IPython; IPython.embed()
                density_outputs[k] = v.view(N, num_z_steps, -1)

            with torch.set_grad_enabled(self.training):
                deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
                alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1)) # [N, T+t]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]

                dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
                for k, v in density_outputs.items():
                    density_outputs[k] = v.view(-1, v.shape[-1])

                mask = weights > 1e-4 # hard coded
                # normals_enc, w_r_enc, n_dot_w_o = self.get_color_mlp_extra_params(normals, dirs) TODO
                rgbs = self.color(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), mask=mask.reshape(-1), **density_outputs)
                rgbs = rgbs.view(N, -1, 3) # [N, T+t, 3]

                #print(xyzs.shape, 'valid_rgb:', mask.sum().item())

                # calculate weight_sum (mask)
                weights_sum = weights.sum(dim=-1) # [N]

            if self.training and use_eikonal_loss:
                n_eik_points = xyzs_eik_near.shape[0]
                xyzs_eik = torch.empty(n_eik_points, 3, device=rays_o.device).uniform_(-self.bound, self.bound)
                xyzs_eik = torch.cat([xyzs_eik, xyzs_eik_near.reshape(-1,3)], 0) if xyzs_eik_near is not None else xyzs_eik
                xyzs_eik = torch.min(torch.max(xyzs_eik, aabb[:3]), aabb[3:]) # a manual clip.
                
                xyzs_eik.requires_grad_(True)
                sdf_eik, *_ = self.forward_sigma(xyzs_eik)
                grad_theta = torch.autograd.grad(sdf_eik, xyzs_eik, torch.ones_like(sdf_eik), retain_graph=True, create_graph=True)[0]
                density_outputs['sdf_gradients'] = grad_theta

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

            weights_sum = weights_sum.new_zeros(prefix).masked_scatter(valid_mask[None,:], weights_sum[None,...])
            image = image.new_zeros(prefix + (3,)).masked_scatter(valid_mask[None,:,None], image[None,...])
            depth = depth.new_zeros(prefix).masked_scatter(valid_mask[None,:], depth[None,...])
            normal_image = normal_image.new_zeros(prefix + (3,)).masked_scatter(valid_mask[None,:,None], normal_image[None,...]) if normal_image is not None else None
        else:
            image = rays_o.new_zeros(prefix + (3,))
            depth = rays_o.new_zeros(prefix)
            normal_image = rays_o.new_zeros(prefix + (3,)) if get_normal_image else None
            weights_sum = rays_o.new_zeros(prefix)

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(sph, rays_d.reshape(-1, 3)) # [N, 3]
        elif bg_color is None:
            bg_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

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