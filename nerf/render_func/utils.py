import torch
import numpy as np

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def curve_plot(x, y, size, ray_idx_map, rays, rays_o, rays_d, xyzs, blend_weight, sdfs, debug_path, xlim=None):
    import matplotlib.pyplot as plt
    idx = x * size + y
    ray_idx = ray_idx_map[idx]
    ray_start = rays[ray_idx, 1].item()
    ray_len = rays[ray_idx, 2].item()
    fig, ax1 = plt.subplots(figsize=(10, 5))
    loc_depth = (xyzs[ray_start:ray_start+ray_len] - rays_o[ray_idx][None,]) / rays_d[ray_idx][None,]
    loc_depth = loc_depth[..., 1].detach()
    
    # add legend and axis labels
    ray_weight = blend_weight[ray_start:ray_start+ray_len]
    ray_sdf = sdfs[ray_start:ray_start+ray_len]
    ax1.plot(loc_depth.cpu(), ray_weight.cpu(), color='blue', label='blend weight')
    ax1.scatter(loc_depth.cpu(), ray_weight.cpu(), color='blue', label='blend weight')
    ax2 = ax1.twinx()
    ax2.plot(loc_depth.cpu(), ray_sdf.cpu(), color='g', label='sdf')
    ax2.scatter(loc_depth.cpu(), ray_sdf.cpu(), color='g', label='sdf')
    # add a line y=0
    ax2.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('sample depth')
    ax1.set_ylabel('blend weight')
    ax2.set_ylabel('sdf')
    # ax1.legend(loc='upper left')
    if xlim is not None: plt.xlim(right=xlim)
    plt.savefig(f'{debug_path}/curve_{x:01d}_{y:01d}.png')
    if x == 20 and y == 20:
        data = np.stack([loc_depth.cpu().numpy(), ray_weight.cpu().numpy(), ray_sdf.cpu().numpy()], axis=1)
        np.savetxt(f'{debug_path}/curve_{x:01d}_{y:01d}.txt', data, fmt='%.6f', delimiter=',')