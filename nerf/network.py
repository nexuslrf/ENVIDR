import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer
from .net_init import init_seq
import numpy as np
import sys
import os

from nerf.sph_loader import METALLIC_THRESHOLD

SQRT3 = 3**0.5

class Density(nn.Module):
    def __init__(self, beta=None):
        super().__init__()
        if beta is not None:
            self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)

class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, beta, beta_min=0.0001, beta_max=1.0):
        super().__init__(beta)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def density_func(self, sdf, beta=None, alpha=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta if alpha is None else alpha
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta_clamp = torch.clamp(self.beta.detach(), self.beta_min, self.beta_max)
        beta_diff = beta_clamp - self.beta.detach()
        beta = self.beta + beta_diff
        # beta = torch.clamp(self.beta, self.beta_min, self.beta_max)
        return beta #* 0 + 0.02

class NeuSDensity(nn.Module):
    def __init__(self, init_val, base_steps=1024, neus_n_detach=False):
        super().__init__()
        self.variance = nn.Parameter(torch.tensor(init_val))
        self.scale = 1.0
        self.base_steps = base_steps
        self.current_steps = base_steps
        self.base_dist = 2*SQRT3 / base_steps
        self.neus_n_detach = neus_n_detach
        
    def update_scale(self, steps):
        if False and steps != self.current_steps:
            self.base_steps = self.current_steps
            self.current_steps = steps
            self.scale = self.current_steps / self.base_steps
            self.base_dist = 2*SQRT3 / self.current_steps
            old_variance = self.variance.data.item()
            self.variance.data = self.variance.data + np.log(self.scale)/10.0
            print(f"Update variance from {old_variance} to {self.variance.data.item()}!")
            
    def get_variance(self):
        return self.variance # + np.log(self.scale)/10.0
    
    def forward(self, sdf, dirs, dists, gradients, cos_anneal_ratio=1.0):
        # sdf, dirs, dists, gradients, cos_anneal_ratio=1.0
        # batch_size, n_samples = sdf.shape[0], sdf.shape[1]
        # if self.scale != 1.0:
        #     variance = self.variance + np.log(self.scale)/10.0
        # else:
        variance = self.variance
        inv_s = torch.exp(variance * 10.0).clip(1e-6, 1e6)   # [1, 1]

        if gradients is not None:
            gradients = gradients.detach() if self.neus_n_detach else gradients 
            true_cos = (dirs * gradients).sum(-1, keepdim=True)

            # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
            # the cos value "not dead" at the beginning training iterations, for better convergence.
            iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                        F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

            # Estimate signed distances at section points
            estimated_next_sdf = sdf + iter_cos.squeeze() * dists * 0.5
            estimated_prev_sdf = sdf - iter_cos.squeeze() * dists * 0.5
        else:
            estimated_next_sdf = sdf - dists * 0.5
            estimated_prev_sdf = sdf + dists * 0.5
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        # alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        return alpha

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 num_levels=16,
                 roughness_bias = -1,
                 opt=None,
                 env_opt=None,
                 **kwargs,
                 ):
        super().__init__(bound, opt=opt, env_opt=env_opt, **kwargs)
        
        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        # if self.opt.eikonal_loss_weight > 0:
        #     encoding = "hashgrid_diff"
        encoding = opt.encoding_pos
        self.encoder, self.in_dim = get_encoder(
                    encoding, level_dim=opt.level_dim,
                    desired_resolution=bound * opt.desired_resolution, 
                    base_resolution=opt.base_resolution,
                    num_levels=num_levels, log2_hashmap_size=opt.log2_hashmap_size,
                    multires=opt.multires # for PE
                )
        self.roughness_bias = roughness_bias


        if self.opt.use_sdf:
            if not self.opt.use_neus_sdf:
                beta_param_init = opt.init_beta
                beta_min, beta_max = opt.beta_min, opt.beta_max
                self.sdf_density = LaplaceDensity(beta_param_init, beta_min, beta_max)
            else:
                print("Using Neus SDF, init variance: ", opt.init_variance)
                self.sdf_density = NeuSDensity(opt.init_variance, opt.max_steps, opt.neus_n_detach)
        
        self.embed_dim = 0
        self.embed_fn = None
        self.geometric_init = opt.geometric_init
        if self.geometric_init:
            inside_outside = opt.inside_outside # True for indoor scenes
            weight_norm = True
            bias = opt.geo_init_bias
        #     multires = 6
        #     from freqencoder import FreqEncoder
        #     self.embed_fn = FreqEncoder(input_dim=3, degree=multires)
        #     self.embed_dim = self.embed_fn.output_dim
        # else:
        #     self.embed_fn = None
        
        self.w_material = False
        self.in_roughness, self.in_metallic, self.in_base_color = 0, 0, 0
        if opt.env_sph_mode:
            self.in_roughness, self.in_metallic, self.in_base_color = \
                int(env_opt.vary_roughness), int(env_opt.vary_metallic), 3*int(env_opt.vary_base_color)
        elif opt.unwrap_env_sphere or opt.render_env_on_sphere:
            self.in_roughness, self.in_metallic, self.in_base_color = 1, 1, 3
        
        material_dims = self.in_roughness + self.in_metallic + self.in_base_color
        self.embed_dim += material_dims
        self.w_material = material_dims > 0

        sdf_net = []
        self.skip_layers = self.opt.skip_layers
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim + self.embed_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim
                if opt.ensemble_mlp:
                    out_dim = out_dim + int(opt.use_roughness) + int(opt.learn_indir_blend)
            elif l+1 in self.skip_layers:
                out_dim = hidden_dim - self.in_dim
            else:
                out_dim = hidden_dim
            lin = nn.Linear(in_dim, out_dim, bias=self.geometric_init or self.opt.mlp_bias)

            if self.geometric_init:
                if l == self.num_layers - 1:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)

                elif self.in_dim > 3 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_layers:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(self.in_dim - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            
                if weight_norm:
                    lin = nn.utils.weight_norm(lin)

            sdf_net.append(lin) # why no bias here?

        self.sdf_net = nn.ModuleList(sdf_net)
        self.sdf_act = nn.ReLU(inplace=True) if not self.geometric_init else nn.Softplus(beta=100)
        
        self.net_act = nn.ReLU(inplace=True)
        if self.opt.color_act == "sigmoid":
            self.color_act = nn.Sigmoid()
        elif self.opt.color_act == "exp":
            self.color_act = trunc_exp
            
        if self.opt.net_init != "" and not self.geometric_init:
            init_seq(self.sdf_net, self.opt.net_init, self.sdf_act)

        if self.opt.use_roughness:
            if not self.opt.ensemble_mlp: 
                self.roughness_layer = nn.Linear(self.geo_feat_dim, 1)
            self.roughness_act = nn.Softplus()

        if self.opt.use_diffuse: # num_layers_diffuse
            self.diffuse_net = []
            in_dim = self.geo_feat_dim
            if self.opt.diffuse_with_env and self.opt.diffuse_env_fusion == "concat":
                    in_dim = self.geo_feat_dim + opt.env_feat_dim
            
            out_dim = self.opt.hidden_dim_diffuse
            for l in range(self.opt.num_layers_diffuse-1):
                self.diffuse_net.append(nn.Linear(in_dim, out_dim, bias=True))
                in_dim = out_dim
            diffuse_dim = 3
            self.diffuse_net.append(nn.Linear(in_dim, diffuse_dim, bias=True))
            self.diffuse_net = nn.ModuleList(self.diffuse_net)
            if self.opt.net_init != "":
                init_seq(self.diffuse_net, self.opt.net_init, self.net_act)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.in_normal_dim, self.in_refdir_dim = 0, 0
        
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir, multires=opt.multires_dir, degree=opt.sh_degree)
        if self.use_normal_with_mlp:
            self.encoder_normal, self.in_normal_dim = get_encoder(encoding_dir, multires=opt.multires_normal, degree=opt.sh_degree)
        if self.use_reflected_dir:
            self.encoder_refdir, self.in_refdir_dim = \
                get_encoder(opt.encoding_ref, multires=opt.multires_refdir, degree=opt.sh_degree)
            self.diffuse_encoder_refdir, self.in_refdir_dim_diffuse = \
                get_encoder(opt.encoding_ref, multires=opt.multires_refdir, degree=opt.sh_degree_diffuse)
            
        # import IPython; IPython.embed()
        self.use_viewdir = not opt.wo_viewdir
        if not self.use_viewdir:
            self.in_dim_dir = 0

        self.use_env_net = self.opt.use_env_net
        # TODO: Note that when we learn a obj., we don't have access to env_opt. and env_nets is meaningless.
        # BTW, you can use opt.env_sph_mode to determine whether to use env_nets.
        if self.use_env_net:
            assert self.use_reflected_dir, "use_env_net requires use_reflected_dir"
            self.use_env_net = True
            def get_env_net(in_dim, out_dim, feat_dim, num_layers, bias=True):
                env_net = []
                for l in range(num_layers-1):
                    env_net.append(nn.Linear(in_dim, out_dim, bias=bias))
                    in_dim = out_dim
                env_net.append(nn.Linear(in_dim, feat_dim, bias=bias))
                env_net = nn.ModuleList(env_net)
                if self.opt.net_init != "":
                    init_seq(env_net, self.opt.net_init, self.net_act)
                return env_net

            self.env_nets, self.env_net = None, None
            if opt.env_sph_mode:
                self.env_nets = nn.ModuleList()
                for _ in range(len(env_opt.env_images_names)):
                    env_net = get_env_net(self.in_refdir_dim, opt.hidden_dim_env, opt.env_feat_dim, opt.num_layers_env, not opt.env_wo_bias)
                    self.env_nets.append(env_net)
            else:
                self.env_net = get_env_net(self.in_refdir_dim, opt.hidden_dim_env, opt.env_feat_dim, opt.num_layers_env, not opt.env_wo_bias)
                if opt.split_diffuse_env:
                    self.diffuse_env_net = get_env_net(self.in_refdir_dim_diffuse, opt.hidden_dim_env_diffuse, opt.env_feat_dim, opt.num_layers_env, not opt.env_wo_bias)

            self.in_refdir_dim = opt.env_feat_dim

            self.renv_net = None
            if self.opt.use_renv:
                # TODO: consider roughness as additional input.
                rgb_dim = 3 # + 3
                roughness_dim = 1
                in_renv_dim = rgb_dim + roughness_dim
                hidden_dim_renv, num_layers_renv = 64, 4
                self.renv_net = get_env_net(in_renv_dim, hidden_dim_renv, opt.env_feat_dim, num_layers_renv, not opt.env_wo_bias)

        color_net =  []

        self.n_dot_viewdir_dim = 1 if self.use_n_dot_viewdir else 0
        # print(f"self.in_normal_dim: {self.in_normal_dim} self.in_dim_dir: {self.in_dim_dir}, self.geo_feat_dim: {self.geo_feat_dim}")
        # print(f"self.in_refdir_dim: {self.in_refdir_dim} self.n_dot_viewdir_dim: {self.n_dot_viewdir_dim}")
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim + self.in_normal_dim + self.in_refdir_dim + self.n_dot_viewdir_dim
            else:
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 3
            else:
                out_dim = hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=self.opt.mlp_bias))

        self.color_net = nn.ModuleList(color_net)
        if self.opt.net_init != "":
            init_seq(self.color_net, self.opt.net_init, self.net_act)
        if self.opt.use_diffuse and self.opt.mlp_bias:
            self.color_net[-1].bias.data -= np.log(3) # make a lower specular at the beginning

        if self.opt.plot_roughness:
            self.input_roughnesses = torch.range(0.0, 1.0, 1.0/15)
            self.low_metal_predicted_roughnesses = []
            self.high_metal_predicted_roughnesses = []
            self.low_metalness = 0.1
            self.high_metalness = 0.9

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            self.encoder_dir_bg, self.in_dim_dir_bg = get_encoder('sphere_harmonics', degree=4)
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir_bg
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
            if self.opt.net_init != "":
                init_seq(self.bg_net, self.opt.net_init, self.net_act)
        else:
            self.bg_net = None

    def concate_material_params(self, x, material):
        # be sure shape matches
        if self.in_roughness:
            # x = torch.cat([x, [material["roughness"]], device=x.device)], dim=-1)
            x = torch.cat([x, material["roughness"] + torch.zeros_like(x[...,:1])], dim=-1)
        if self.in_metallic:
            x = torch.cat([x, material["metallic"] + torch.zeros_like(x[...,:1])], dim=-1)
        if self.in_base_color:
            color = torch.tensor(material["color"][:3], dtype=x.dtype, device=x.device) # [3]
            x = torch.cat([x, color + torch.zeros_like(x[...,:3])], dim=-1)
        return x

    def forward_geometry(self, xyz, material=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # material: dict
        # print("calling forward", file=sys.stderr)

        # sigma
        x = self.encoder(xyz, bound=self.bound)
        # Ruofan's exp
        if self.opt.enabled_levels > 0:
            mask = torch.zeros(size=(self.opt.num_levels, 2), device=x.device)
            mask[:self.opt.enabled_levels, :]  += 1
            x = x * mask.reshape(-1)

        if self.embed_fn is not None:
            x = torch.cat([self.embed_fn(xyz), x], dim=-1)

        # another question: where to concatenate?
        # 1. after encoder [√]
        # 2. after geofeat [x] [not tested yet]
        
        if self.opt.plot_roughness:
            if material["metallic"] > METALLIC_THRESHOLD:
                if len(self.high_metal_predicted_roughnesses) < self.input_roughnesses.shape[0]:
                    material["metallic"] = self.high_metalness
                    material["roughness"] = self.input_roughnesses[len(self.high_metal_predicted_roughnesses)]
            else:
                if len(self.low_metal_predicted_roughnesses) < self.input_roughnesses.shape[0]:
                    material["metallic"] = self.low_metalness
                    material["roughness"] = self.input_roughnesses[len(self.low_metal_predicted_roughnesses)]

        if self.w_material:
            x = self.concate_material_params(x, material)

        h = x
        for l in range(self.num_layers):
            if l in self.skip_layers:
                h = torch.cat([h, x], dim=-1) / np.sqrt(2)
            h = self.sdf_net[l](h)
            if l != self.num_layers - 1:
                h = self.sdf_act(h)

        #sigma = F.relu(h[..., 0])
        if self.use_sdf:
            sdf = h[..., 0]
            sigma = None
        else:
            sdf = None
            sigma = trunc_exp(h[..., 0])
       
        geo_feat = h[..., 1:1+self.geo_feat_dim]
        if self.opt.geo_feat_act == "tanh":
            geo_feat = torch.tanh(geo_feat)
        elif self.opt.geo_feat_act == "unitNorm":
            geo_feat = F.normalize(geo_feat, dim=-1)
        elif self.opt.geo_feat_act == 'instanceNorm':
            geo_feat_mean = geo_feat.mean(dim=-1, keepdim=True)
            geo_feat_var = geo_feat.var(dim=-1, keepdim=True)
            eps = 1e-5
            geo_feat = (geo_feat - geo_feat_mean) / torch.sqrt(geo_feat_var + eps)

        if self.opt.use_roughness and not self.opt.diffuse_only and not self.opt.bypass_roughness:
            raw_roughness = self.roughness_layer(geo_feat) if not self.opt.ensemble_mlp \
                    else h[..., 1+self.geo_feat_dim:2+self.geo_feat_dim]
            if self.opt.learn_indir_blend and self.opt.ensemble_mlp:
                self.blend_weight = torch.sigmoid(h[..., 2+self.geo_feat_dim:3+self.geo_feat_dim])
            self.roughness = self.opt.roughness_act_scale * self.roughness_act(raw_roughness + self.roughness_bias)
            self.roughness = self.roughness * self.opt.roughness_scale
        else:
            self.roughness = self.opt.default_roughness

        if self.opt.plot_roughness:
            sample_size = 4
            start_index = int(self.roughness.shape[1] / 2) - int(sample_size / 2) # 6
            if material["metallic"] > METALLIC_THRESHOLD:
                if len(self.high_metal_predicted_roughnesses) < self.input_roughnesses.shape[0]:
                    roughness_value = torch.sum(self.roughness[:, start_index: start_index + sample_size, :], dim=1) / sample_size
                    roughness_value = torch.mean(roughness_value)
                    self.high_metal_predicted_roughnesses.append(roughness_value.cpu().item())
                    # self.high_metal_predicted_roughnesses.append(torch.mean(self.roughness).cpu().item())
            else:
                if len(self.low_metal_predicted_roughnesses) < self.input_roughnesses.shape[0]:
                    roughness_value = torch.sum(self.roughness[:, start_index: start_index + sample_size, :], dim=1) / sample_size
                    roughness_value = torch.mean(roughness_value)
                    self.low_metal_predicted_roughnesses.append(roughness_value.cpu().item())
            if len(self.high_metal_predicted_roughnesses) == self.input_roughnesses.shape[0] \
                and len(self.low_metal_predicted_roughnesses) == self.input_roughnesses.shape[0]:
                import matplotlib.pyplot as plt
                self.input_roughnesses = self.input_roughnesses.cpu().numpy()
                plt.plot(self.input_roughnesses, np.array(self.low_metal_predicted_roughnesses),  label =f"m={self.low_metalness}")
                plt.plot(self.input_roughnesses, np.array(self.high_metal_predicted_roughnesses),  label =f"m={self.high_metalness}")
                
                import numpy.polynomial.polynomial as poly
                
                def add_poly_fit_curve(x, y, degree, label):
                    coefs = poly.polyfit(x, y, degree)
                    ffit = poly.polyval(x, coefs)
                    plt.plot(x, ffit, label=label)
                    return coefs

                low_metal_coefs = add_poly_fit_curve(self.input_roughnesses, np.array(self.low_metal_predicted_roughnesses), 3, f"m={self.low_metalness} poly_fitted")
                high_metal_coefs = add_poly_fit_curve(self.input_roughnesses, np.array(self.high_metal_predicted_roughnesses), 3, f"m={self.high_metalness} poly_fitted")
                print(f"\nlow_metal_coefs {low_metal_coefs}")
                print(f"high_metal_coefs {high_metal_coefs}")

                plt.xlabel("input roughness")
                plt.ylabel("predicted roughness")
                plt.legend()
                plt.title('average roughness')
                plt.savefig(os.path.join(self.opt.workspace, "roughness.png"))
                import IPython; IPython.embed()

        self.metallic = 1.

        return sdf, sigma, geo_feat   

    def forward_sigma(self, xyzs, material=None, **kwargs):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # print("calling forward", file=sys.stderr)
        use_sdf_sigma_grad = kwargs.get("use_sdf_sigma_grad", False)

        sdfs, sigmas, geo_feats = self.forward_geometry(xyzs, material)
        normals = None
        eikonal_sdf_gradients = None
        if self.use_sdf:
            if use_sdf_sigma_grad:
                # compute normal
                normals, eikonal_sdf_gradients = self.compute_normal(sdfs, xyzs, self.opt.eikonal_loss)
            # compute sdfs -> sigmas
            assert sigmas is None
            # dirs, dists, gradients, cos_anneal_ratio=1.0
            if self.opt.use_neus_sdf:
                dirs = kwargs.get("dirs", None)
                dists = kwargs.get("dists", 2 * SQRT3 / self.sdf_density.base_steps)
                sigmas = self.sdf_density(sdfs, dirs=dirs, dists=dists, gradients=normals, cos_anneal_ratio=self.opt.cos_anneal_ratio)
            else:
                sigmas = self.sdf_density(sdfs)
        elif (not self.use_sdf) and use_sdf_sigma_grad:
            normals, eikonal_sdf_gradients = self.compute_normal(sigmas, xyzs, self.opt.eikonal_loss)
        
        return sdfs, sigmas, geo_feats, normals, eikonal_sdf_gradients

    def forward_color(self, geo_feat, d, normal=None, w_r=None, n_dot_w_o=None, use_specular_color=False, env_net_index=0, n_env_enc=None, r_images=None, roughness=None):

        # color
        if self.opt.use_diffuse:
            h = geo_feat 
            if self.opt.diffuse_with_env:
                env_net = self.env_nets[env_net_index] if self.opt.env_sph_mode else self.env_net
                if self.opt.split_diffuse_env:
                    env_net = self.diffuse_env_net
                for l in range(self.opt.num_layers_env):
                    n_env_enc = env_net[l](n_env_enc)
                    if l != self.opt.num_layers_env - 1:
                        n_env_enc = self.net_act(n_env_enc)

                if self.opt.env_feat_act == 'tanh':
                    n_env_enc = torch.tanh(n_env_enc)
                elif self.opt.env_feat_act == 'unitNorm':
                    n_env_enc = F.normalize(n_env_enc, dim=-1)
                elif self.opt.env_feat_act == 'instanceNorm':
                    n_env_enc_mean = n_env_enc.mean(dim=-1, keepdim=True)
                    n_env_enc_var = n_env_enc.var(dim=-1, keepdim=True)
                    eps = 1e-5
                    n_env_enc = (n_env_enc - n_env_enc_mean) / torch.sqrt(n_env_enc_var + eps)
                    
                if self.opt.diffuse_env_fusion == "concat":
                    h = torch.cat([h, n_env_enc], dim=-1)
                elif self.opt.diffuse_env_fusion == "add":
                    h = h + n_env_enc                    
                elif self.opt.diffuse_env_fusion == "mul":
                    h = h * n_env_enc
            
            for l in range(self.opt.num_layers_diffuse):
                h = self.diffuse_net[l](h)
                if l != self.opt.num_layers_diffuse - 1:
                    h = self.net_act(h)

            if self.opt.diffuse_env_fusion == "rgb":
                for i, lin in enumerate(self.diffuse_env_decoder):
                    n_env_enc = self.net_act(lin(n_env_enc)) if i != len(self.diffuse_env_decoder) - 1 else lin(n_env_enc)
                # n_env_enc = F.softplus(n_env_enc) #torch.exp(0.5 * n_env_enc.clamp(-10,10)) # hdr act
                # self.base_color = self.c_diffuse
                _shape = h.shape
                self.c_diffuse = self.color_act((n_env_enc * h).reshape(*_shape[:-1], 3, -1).sum(-1))
            else:
                self.c_diffuse = self.color_act(h)      
            if getattr(self, 'metallic') is not None:
                self.c_diffuse = self.c_diffuse * self.metallic   
        else:
            self.c_diffuse = 0

        if not self.opt.diffuse_only:
            # print("not using diffuse only", file=sys.stderr)
            if self.use_viewdir:
                d = self.encoder_dir(d)
                h = torch.cat([d, geo_feat], dim=-1)
            else:
                h = geo_feat

            if self.use_normal_with_mlp: 
                assert normal is not None
                h = torch.cat([h, normal], dim=-1)
            
            branch_dict = {}
            renv_mask, blend_weight = None, 1
            if w_r is not None and not self.opt.train_renv:
                if self.use_env_net:
                    env_net = self.env_nets[env_net_index] if self.opt.env_sph_mode else self.env_net

                    for l in range(self.opt.num_layers_env):
                        w_r = env_net[l](w_r)
                        if l != self.opt.num_layers_env - 1:
                            w_r = self.net_act(w_r)
                    
                    if self.opt.env_feat_act == 'tanh':
                        w_r = torch.tanh(w_r)
                    elif self.opt.env_feat_act == 'unitNorm':
                        w_r = F.normalize(w_r, dim=-1)
                    elif self.opt.env_feat_act == 'instanceNorm':
                        w_r_mean = w_r.mean(dim=-1, keepdim=True)
                        w_r_var = w_r.var(dim=-1, keepdim=True)
                        eps = 1e-5
                        w_r = (w_r - w_r_mean) / torch.sqrt(w_r_var + eps)

                    h_env = torch.cat([h, w_r], dim=-1)
                else:
                    h_env = torch.cat([h, w_r], dim=-1)
                branch_dict['env'] = h_env

            if r_images is not None and self.opt.use_renv:
                # roughness_remap = 0.99 * torch.ones_like(roughness) 
                # renv_roughness = roughness / self.opt.roughness_scale 
                if not self.opt.train_renv:
                    ide_roughness_thresh = self.opt.indir_roughness_thresh
                    renv_mask = roughness.squeeze() < ide_roughness_thresh
                    if r_images.shape[-1] == 4:
                        r_vis = r_images[..., -1]
                        r_images = r_images[..., :3] * r_vis[..., None].detach()
                        renv_mask = renv_mask * (r_vis > 0.9)
                    r_images = r_images[renv_mask]
                    _roughness = roughness[renv_mask] / self.opt.roughness_scale # better keep it unscaled
                    _h = h[renv_mask]
                    roughness_remap = torch.sqrt(_roughness / 0.75) #.detach()
                    # if not self.opt.grad_rays:
                    #     roughness_remap = roughness_remap.detach()
                    # TODO: add scale here
                    if not self.opt.learn_indir_blend:
                        linear_roughness_thresh = 0.18 #if not self.opt.rf_debug else 0.18 #0.6 #0.15
                        blend_weight = 0.95 * torch.sigmoid(80 * (roughness_remap - linear_roughness_thresh))
                    else:
                        blend_weight = 0.98 * self.blend_weight[renv_mask]
                else:
                    _h = h
                    roughness_remap = torch.sqrt(roughness / 0.75)

                renv_enc = torch.cat([r_images, roughness_remap], dim=-1)
                # renv_enc = torch.cat([r_images, r_images * (1-roughness_remap.clamp_max(0.99)), roughness_remap], dim=-1)
                # renv_enc = torch.cat([r_images * (1-roughness_remap.clamp_max(0.99)), roughness_remap], dim=-1)
                for l, lin in enumerate(self.renv_net):
                    renv_enc = lin(renv_enc)
                    if l != self.opt.num_layers_env - 1:
                        renv_enc = self.net_act(renv_enc)
                
                if self.opt.env_feat_act == 'tanh':
                    renv_enc = torch.tanh(renv_enc)
                elif self.opt.env_feat_act == 'unitNorm':
                    renv_enc = F.normalize(renv_enc, dim=-1)
                elif self.opt.env_feat_act == 'instanceNorm':
                    renv_enc_mean = renv_enc.mean(dim=-1, keepdim=True)
                    renv_enc_var = renv_enc.var(dim=-1, keepdim=True)
                    eps = 1e-5
                    renv_enc = (renv_enc - renv_enc_mean) / torch.sqrt(renv_enc_var + eps)
                    
                # TODO: is this shape correct?
                renv_enc = renv_enc * torch.ones_like(_h[...,:1])
                h_renv = torch.cat([_h, renv_enc], dim=-1)
                branch_dict['renv'] = h_renv

            if not branch_dict:
                branch_dict['env'] = h
            
            color_dict = {}
            for k, h_c in branch_dict.items():
                if n_dot_w_o is not None:
                    if k == 'renv' and renv_mask is not None:
                        h_c = torch.cat([h_c, n_dot_w_o[renv_mask]], dim=-1)
                    else:
                        h_c = torch.cat([h_c, n_dot_w_o], dim=-1)

                for l in range(self.num_layers_color):
                    h_c = self.color_net[l](h_c)
                    if l != self.num_layers_color - 1:
                        h_c = self.net_act(h_c)

                # sigmoid activation for rgb
                c_specular = self.color_act(h_c)
                color_dict[k] = c_specular
            
            
            self.c_specular = color_dict['env'] if not self.opt.train_renv else color_dict['renv']
            if 'renv' in color_dict:
                if self.opt.indir_only:
                    self.c_specular = self.c_specular * 0
                    # blend_weight = blend_weight * 0
                if renv_mask is not None:
                    self.c_specular = self.c_specular.masked_scatter(renv_mask[:, None], self.c_specular[renv_mask] * blend_weight + color_dict['renv'] * (1-blend_weight))
                else:
                    self.c_specular = self.c_specular * blend_weight + color_dict['renv'] * (1-blend_weight)
            
        else:
            self.c_specular = 0

        color = (self.c_diffuse + self.c_specular) * self.opt.intensity_scale


        return color        

    def forward(self, x, d, normal=None, w_r=None, n_dot_w_o=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # print("calling forward", file=sys.stderr)

        # sigma
        sdf, sigma, geo_feat = self.forward_sigma(x)

        # color
        color = self.forward_color(geo_feat, d, normal, w_r, n_dot_w_o)

        return sdf, sigma, color

    def density(self, x, **kwargs):
        # x: [N, 3], in [-bound, bound]
        # print("calling density", file=sys.stderr)

        sdf, sigma, geo_feat, normal, eikonal_sdf_gradient = self.forward_sigma(x, **kwargs)

        return {
            'sdf': sdf,
            'sigma': sigma,
            'geo_feat': geo_feat,
            'normal': normal,
            'sdf_gradients': eikonal_sdf_gradient
        }

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir_bg(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = self.color_act(h)

        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, normal=None, w_r=None, n_dot_w_o=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

            normal = normal[mask] if normal is not None else None
            w_r = w_r[mask] if w_r is not None else None
            n_dot_w_o = n_dot_w_o[mask] if n_dot_w_o is not None else None

        h = self.forward_color(geo_feat, d, normal, w_r, n_dot_w_o)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs        

    # optimizer utils
    def get_params(self, lr, plr=0, slr=0, elr=0):
        plr = lr if plr == 0 else plr
        slr = lr if slr == 0 else slr
        elr = lr if elr == 0 else elr
        if not self.opt.train_renv:
            params = []
            if not self.opt.train_env_only:
                params = [
                    {'params': self.encoder.parameters(), 'lr': plr},
                    {'params': self.sdf_net.parameters(), 'lr': lr},
                    # {'params': self.encoder_dir.parameters(), 'lr': lr},
                ]
                # print(f"self.opt.freeze_specular_mlp={self.opt.freeze_specular_mlp}")
                if 'specular' not in self.opt.frozen_mlps:
                    params.append(
                        {'params': self.color_net.parameters(), 'lr': lr},
                    )
                if self.use_sdf:
                    params.append(
                        {'params': self.sdf_density.parameters(), 'lr': slr},
                    )
                if self.opt.use_diffuse and 'diffuse' not in self.opt.frozen_mlps:
                    params.append(
                        {'params': self.diffuse_net.parameters(), 'lr': lr},
                    )

                # import IPython; IPython.embed()
                if self.bg_radius > 0:
                    params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
                    params.append({'params': self.bg_net.parameters(), 'lr': lr})
                
                if self.opt.use_roughness and not self.opt.ensemble_mlp: 
                    params.append({'params': self.roughness_layer.parameters(), 'lr': lr})

            if self.use_env_net:
                if self.env_net is not None:
                    params.append({'params': self.env_net.parameters(), 'lr': elr})
                elif self.env_nets is not None:
                    params.append({'params': self.env_nets.parameters(), 'lr': elr})

                if self.renv_net is not None and 'renv' not in self.opt.frozen_mlps:
                    params.append({'params': self.renv_net.parameters(), 'lr': lr})                
        else:
            # Train renv only
            params = []
            params.append({'params': self.renv_net.parameters(), 'lr': lr})

        return params
