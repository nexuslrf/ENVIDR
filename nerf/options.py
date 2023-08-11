import configargparse
import numpy as np

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--plr', type=float, default=0, help="initial learning rate for grid, if 0, use lr")
    parser.add_argument('--slr', type=float, default=0, help="initial learning rate for scalar, if 0, use lr")
    parser.add_argument('--elr', type=float, default=0, help="initial learning rate for env, if 0, use lr")
    parser.add_argument('--ckpt_model_only', action='store_true')
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--num_rays_sched', type=int, default=[], nargs='+', help="[num_rays_1, iters_1, ...]")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--max_steps_sched', type=int, default=[], nargs='+', help="[steps_1, iters_1, ...] (only valid when using --cuda_ray)")
    parser.add_argument('--early_stop_steps', type=int, default=-1, help="stop sample after this steps per ray (only valid when using --cuda_ray)")
    parser.add_argument('--early_stop_steps_sched', type=int, default=[], nargs='+', help="[steps_1, iters_1, ...] (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--update_extra_before', type=int, default=-1, help="iter to stop update extra status (only valid when using --cuda_ray and > 0)")
    parser.add_argument('--extra_state_full_update', action='store_true', help="reset & update full extra state")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--max_ray_batch_cuda', type=int, default=-1, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")
    parser.add_argument('--image_batch', type=int, default=1, help="batch size of images at inference")
    parser.add_argument('--max_keep_ckpt', type=int, default=2)
    parser.add_argument('--ckpt_step', type=int, default=0)
    parser.add_argument('--T_thresh', type=float, default=1e-4)
    parser.add_argument('--stratified_sampling', action='store_true', help="use stratified sampling")
    
    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")
    parser.add_argument('--encoding_pos', type=str, default='hashgrid', choices=['hashgrid', 'hashgrid_diff', 'frequency'])
    parser.add_argument('--num_levels', type=int, default=16)
    parser.add_argument('--level_dim', type=int, default=2)
    parser.add_argument('--enabled_levels', type=int, default=-1)
    parser.add_argument('--enabled_levels_sched', type=int, default=[], nargs='+', help="[enabled_levels_1, iters_1]")
    parser.add_argument('--base_resolution', type=int, default=16)
    parser.add_argument('--desired_resolution', type=int, default=2048)
    parser.add_argument('--log2_hashmap_size', type=int, default=19)
    ### geo network options
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2, help="num layers in MLP")
    parser.add_argument('--skip_layers', type=int, default=[], nargs='+', help="skip layers in MLP")
    parser.add_argument('--multires', type=int, default=6, help="multiresolution factor for PE")
    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)", choices=['linear', 'srgb', 'hdr'])
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    parser.add_argument('--blur_sigma', type=int, default=0, help="sigma of gaussian blur for input image")

    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--num_layers_bg', type=int, default=2, help="num layers in MLP for background model")
    parser.add_argument('--marching_aabb', type=float, nargs='*', default=[], help="if provided, use marching_aabb to replace bounds in ray marching")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--center_crop', type=float, default=0, help="center crop factor of the input image")
    parser.add_argument('--center_crop_ratio', type=float, default=0.6, help="the ratio of center crop sample of N samples")
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--error_map_start_iter', type=int, default=0, help="error_map_start_iter")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    # new options
    parser.add_argument('--use_sdf', action='store_true', help="use sdf to predict density (as in VolNerf)")
    parser.add_argument('--use_neus_sdf', action='store_true', help="use sdf to predict density (as in NeuS)")
    parser.add_argument('--init_variance', type=float, default=0.3)
    parser.add_argument('--neus_n_detach', action='store_true', help="rescale alpha to [0, 1]")
    parser.add_argument('--geo_init_bias', type=float, default=1.0)
    parser.add_argument('--inside_outside', action='store_true', help="use inside_outside")
    parser.add_argument('--cos_anneal_steps', type=int, default=5000)
    parser.add_argument('--cos_anneal_ratio', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=0)
    
    parser.add_argument('--relsdf_loss_start_iter', type=int, default=0, help="relsdf_loss_start_iter")
    parser.add_argument('--relsdf_loss_weight', type=float, default=0, help="relsdf_loss_weight") # 5e-3
    parser.add_argument('--relsdf_loss_weight_sched', type=float, default=[], nargs='+', help='[start, end, epoches, stop], stop=-1 won\'t stop')
    parser.add_argument('--relsdf_mode', type=str, default='mean', choices=['mean', 'sum'])
    parser.add_argument('--dist_bound_weight', type=float, default=0) 
    parser.add_argument('--dist_bound_start_iter', type=int, default=0) 
    # parser.add_argument('--relsdf_loss', action='store_true', help="use relsdf loss")
    parser.add_argument('--backsdf_loss_start_iter', type=int, default=0, help="backsdf_loss_start_iter")
    parser.add_argument('--backsdf_loss_weight', type=float, default=0, help="backsdf_loss_weight") # 5e-3
    parser.add_argument('--backsdf_thresh', type=float, default=0.1, help="backsdf_thresh") # 0.1
    parser.add_argument('--backsdf_mode', type=str, default='sum', choices=['sum', 'mean'], help="backsdf_mode")
    parser.add_argument('--backsdf_loss_weight_sched', type=float, default=[], nargs='+', help='[start, end, epoches, stop], stop=-1 won\'t stop')
    # parser.add_argument('--backsdf_loss', action='store_true', help="use backsdf loss")
    parser.add_argument('--cauchy_loss_start_iter', type=int, default=0, help="cauchy_loss_start_iter")
    parser.add_argument('--cauchy_loss_weight', type=float, default=0, help="cauchy_loss_weight") # 5e-3
    parser.add_argument('--cauchy_roughness_weighted', action='store_true', help="use roughness weighted cauchy loss")
    parser.add_argument('--cauchy_undetach_beta', action='store_true')
    parser.add_argument('--cauchy_loss_weight_sched', type=float, default=[], nargs='+', help='[start, end, epoches, stop], stop=-1 won\'t stop')
    # parser.add_argument('--cauchy_loss', action='store_true', help="use cauchy loss")
    parser.add_argument('--weighted_eikonal', action='store_true', help="use weighted eikonal loss")
    parser.add_argument('--eikonal_loss_start_iter', type=int, default=0, help="eikonal_loss_start_iter")
    parser.add_argument('--eikonal_loss_weight', type=float, default=0, help="eikonal_loss_weight")
    parser.add_argument('--eikonal_loss_weight_sched', type=float, default=[], nargs='+', help='[start, end, epoches, stop], stop=-1 won\'t stop')
    # parser.add_argument('--eikonal_loss', action='store_true', help="use eikonal loss")
    parser.add_argument('--sdf_loss_weight', type=float, default=0, help="sdf_loss_weight")

    parser.add_argument('--orientation_loss_weight', type=float, default=0, help="orientation_loss_weight")
    parser.add_argument('--orientation_loss_start_iter', type=int, default=0, help="orientation_loss_start_iter")
    parser.add_argument('--orientation_loss_weight_sched', type=float, default=[], nargs='+', help='[start, end, epoches, stop], stop=-1 won\'t stop')

    parser.add_argument('--entropy_loss_weight', type=float, default=0, help="entropy_loss_weight")
    # 
    parser.add_argument('--mask_loss_weight', type=float, default=0, help="mask_loss_weight")
    parser.add_argument('--mask_loss_start_iter', type=int, default=0, help="mask_loss_start_iter")

    parser.add_argument('--color_loss', type=str, default='l2', choices=['l1', 'l2', 'huber', 'relativel2'])
    parser.add_argument('--color_l1_loss', action='store_true', help="use L1 loss for color")
    parser.add_argument('--color_loss_weight', type=float, default=1.)

    parser.add_argument('--diffuse_loss', action='store_true', help="use separate diffuse loss for color")
    parser.add_argument('--diffuse_loss_weight', type=float, default=1.)

    parser.add_argument('--debug', action='store_true', help="debug mode")
    parser.add_argument('--debug_id', type=int, default=10)
    parser.add_argument('--debug_patch_h', type=int, default=0)
    parser.add_argument('--debug_patch_w', type=int, default=0)
    parser.add_argument('--debug_patch_size', type=int, default=50)
    parser.add_argument('--debug_patch_step', type=int, default=10)
    parser.add_argument('--debug_stop_epoch', type=int, default=-1)

    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'val', 'train'], help="test split")
    parser.add_argument('--test_ids', type=int, default=[], nargs='+', help="test ids")
    parser.add_argument('--test_skip', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=200, help="eval_interval")

    # color mlp cfg
    parser.add_argument('--color_act', type=str, default='sigmoid', choices=['sigmoid', 'exp'])
    parser.add_argument('--num_layers_color', type=int, default=3, help="num layers in MLP for foreground")
    parser.add_argument('--hidden_dim_color', type=int, default=64)
    parser.add_argument('--encoding_dir', type=str, default="sphere_harmonics", help="encoding direction", choices=["sphere_harmonics", "frequency"])
    parser.add_argument('--multires_dir', type=int, default=0, help="num freqs in encoding")

    parser.add_argument('--detach_normal', action='store_true', help="detach normal")
    
    parser.add_argument('--normal_with_mlp', action='store_true', help="use normal MLP")
    parser.add_argument('--multires_normal', type=int, default=0, help="num freqs in encoding for normal")

    parser.add_argument('--sh_degree', type=int, default=4, help="sh degree")


    parser.add_argument('--error_bound_sample', action='store_true', help="use error bound sample")
    parser.add_argument('--error_bound_start_iter', type=int, default=0)
    parser.add_argument('--geometric_init', action='store_true', help="use geometric init")

    parser.add_argument('--init_beta', type=float, default=0.1)
    parser.add_argument('--beta_min', type=float, default=0.0001)
    parser.add_argument('--beta_max', type=float, default=1)
    parser.add_argument('--beta_min_sched', type=float, default=[], nargs='+', help='[bm_s, bm_e, iters]')

    parser.add_argument('--render_bg_color', type=str, default='white', choices=['white', 'black'])
    parser.add_argument('--alpha_bg_mode', type=str, choices=['white', 'random'], default='random')
    parser.add_argument('--net_activation', type=str, choices=['relu', 'leaky_relu', 'softplus'], default='relu')
    parser.add_argument('--net_init', type=str, default='')
    parser.add_argument('--mlp_bias', action='store_true', help="use bias in MLP")

    parser.add_argument('--geo_feat_act', type=str, choices=['', 'tanh', 'unitNorm', 'instanceNorm'], default='')
    parser.add_argument('--env_feat_act', type=str, choices=['', 'tanh', 'unitNorm', 'instanceNorm'], default='')

    # more for dealing with specular
    parser.add_argument('--use_diffuse', action='store_true', help="use diffuse")
    parser.add_argument('--diffuse_only', action='store_true', help="use diffuse only")
    parser.add_argument('--color_net_start_iter', type=int, default=0, help="color_net_start_iter")
    parser.add_argument('--num_layers_diffuse', type=int, default=2, help="num layers in MLP for diffuse")
    parser.add_argument('--hidden_dim_diffuse', type=int, default=32, help="num layers in MLP for diffuse")
    parser.add_argument('--diffuse_with_env', action='store_true', help="use env for diffuse")

    parser.add_argument('--diffuse_env_fusion', type=str, default='concat', choices=['concat', 'add', 'mul'])
    parser.add_argument('--specular_env_fusion', type=str, default='concat', choices=['concat'])
    
    parser.add_argument('--visual_items', type=str, nargs='+', default=[])

    # ref dir
    parser.add_argument('--use_reflected_dir', action='store_true', help="use reflected_dir with color MLP")
    parser.add_argument('--multires_refdir', type=int, default=0, help="num freqs in encoding for reflected_dir")
    parser.add_argument('--use_n_dot_viewdir', action='store_true', help="use n_dot_viewdir with color MLP")

    parser.add_argument('--in_normal_mode', type=str, choices=['point', 'surface'], default='surface')
    parser.add_argument('--use_env_net', action='store_true', help="use env net")
    parser.add_argument('--env_fuse_mode', type=str, default='concat', choices=['concat', 'add', 'mul'])
    parser.add_argument('--num_layers_env', type=int, default=4, help="num layers in MLP for env")
    parser.add_argument('--hidden_dim_env', type=int, default=128)
    parser.add_argument('--env_feat_dim', type=int, default=16)
    parser.add_argument('--env_wo_bias', action='store_true')
    parser.add_argument('--geo_feat_dim', type=int, default=15)
    parser.add_argument('--encoding_ref', type=str, default="frequency", help="encoding direction", 
                        choices=["sphere_harmonics", "frequency", "integrated_dir", "integrated_dir_real"])

    parser.add_argument('--wo_viewdir', action='store_true', help="wo_viewdir")
    
    parser.add_argument('--normal_anneal_iters', type=int, default=0, help="normal_anneal_iters")
    parser.add_argument('--normal_anneal_ratio', type=float, default=1, help="normal_anneal_iters")

    parser.add_argument('--beta_loss_start_iter', type=int, default=0)
    parser.add_argument('--beta_loss_weight', type=float, default=0)

    # for lighting
    parser.add_argument('--swap_env_path', type=str, default='')
    parser.add_argument('--sph_renderer', type=str, default='filament', choices=['filament', 'mitsuba'])
    parser.add_argument('--env_sph_mode', action='store_true', help="train env sphere ball")
    parser.add_argument('--env_sph_radius', type=float, default=1)
    parser.add_argument('--env_dataset_config', type=str, default='')
    parser.add_argument('--light_intensity_scale', type=float, default=1)

    parser.add_argument('--color_mlp_path', type=str, default='')
    parser.add_argument('--frozen_mlps', type=str, nargs='+', default=[], choices=['specular', 'diffuse', 'diffuse_env', 'specular_env', 'renv']) 
    parser.add_argument('--resume_mlps', type=str, nargs='+', default=[], choices=['specular', 'diffuse', 'diffuse_env', 'specular_env', 'renv']) 

    parser.add_argument('--use_roughness', action='store_true', help="use roughness")
    parser.add_argument('--diffuse_kappa_inv', type=float, default=0.64, help="diffuse_kappa_inv")
    parser.add_argument('--default_roughness', type=float, default=0.05, help="default_roughness")
    parser.add_argument('--split_diffuse_env', action='store_true', help="split diffuse env")
    parser.add_argument('--hidden_dim_env_diffuse', type=int, default=-1, help="hidden_dim_env_diffuse")
    parser.add_argument('--sh_degree_diffuse', type=int, default=-1, help="sh_degree")
    parser.add_argument('--roughness_scale', type=float, default=1, help="roughness_scale")
    parser.add_argument('--roughness_act_scale', type=float, default=0.2, help="roughness_scale")

    parser.add_argument('--ensemble_mlp', action='store_true', help="")
    
    parser.add_argument('--indir_ref', action='store_true')
    parser.add_argument('--dir_only', action='store_true')
    parser.add_argument('--indir_only', action='store_true')
    parser.add_argument('--indir_ref_start_iter', type=int, default=-1)
    parser.add_argument('--indir_roughness_thresh', type=float, default=0.1)
    parser.add_argument('--obj_aabb', type=float, nargs='*', default=None, help="obj_aabb, p min, p max, 6dim")
    parser.add_argument('--indir_early_stop_steps', type=int, default=32)
    parser.add_argument('--indir_max_steps', type=int, default=1024)
    parser.add_argument('--learn_indir_blend', action='store_true')
    parser.add_argument('--grad_rays', action='store_true')
    parser.add_argument('--grad_rays_scale', type=float, default=0.01)
    parser.add_argument('--grad_rays_start_iter', type=int, default=100)
    parser.add_argument('--plot_roughness', action='store_true')

    parser.add_argument('--train_renv', action="store_true")
    parser.add_argument('--use_renv', action='store_true')
    parser.add_argument('--renv_mlp_path', type=str, default='')

    parser.add_argument('--rf_debug', action='store_true')

    parser.add_argument('--unwrap_env_sphere', action='store_true')
    parser.add_argument('--unwrap_env_id', type=int)
    parser.add_argument('--img_format', type=str, default='jpg', choices=['jpg', 'png', 'exr'])
    parser.add_argument('--unwrap_roughness', type=float, default=0.7)
    parser.add_argument('--unwrap_metallic', type=float, default=0.9)
    parser.add_argument('--unwrap_color', type=float, default=[0.7, 0.7, 0.7], nargs='*')
    parser.add_argument('--unwrap_color_intensity', type=float, default=1.0)

    parser.add_argument('--intensity_scale', type=float, default=1)
    parser.add_argument("--val_folder_name", type=str, default="validation", help="validation folder name")
    parser.add_argument("--render_env_on_sphere", action="store_true", help="render env on sphere")
    
    parser.add_argument("--overwrite_materials", action="store_true", help="overwrite_materials (only for eval)")
    parser.add_argument('--set_env_net_index', type=int, default=-1, help="set_env_net_index (only for eval)")
    parser.add_argument("--train_env_only", action='store_true')
    
    parser.add_argument('--env_rot_degree_range', type=int, default=[], nargs='+', help="[degree_start, degree_end, num_views]")
    
    opt = parser.parse_args()

    opt.relsdf_loss = False
    opt.backsdf_loss = False
    opt.cauchy_loss = False
    opt.eikonal_loss = False
    opt.beta_loss = False
    opt.orientation_loss = False
    opt.dist_bound = False
    opt.bypass_roughness = False
    opt.mask_loss = False
    
    if opt.diffuse_loss_weight > 0:
        opt.diffuse_loss = True
        
    opt.offset = np.asarray(opt.offset)
    
    if opt.hidden_dim_env_diffuse < 0:
        opt.hidden_dim_env_diffuse = opt.hidden_dim_env
    
    if opt.sh_degree_diffuse < 0:
        opt.sh_degree_diffuse = opt.sh_degree
    
    if opt.color_l1_loss:
        print('--color_l1_loss no longer used! switch to `--color_loss=l1`')
        opt.color_loss = 'l1'
        # exit()

    # if opt.cuda_ray:
    #     assert not opt.use_neus_sdf, "cuda ray does not support neus sdf"

    if opt.stratified_sampling:
        assert opt.dt_gamma == 0, "stratified sampling does not support dt_gamma"
    
    if opt.use_neus_sdf:
        opt.use_sdf = True

    if opt.env_sph_mode:
        opt.cuda_ray = False

    if opt.train_renv:
        opt.use_env_net = True
        opt.use_renv = True

    if opt.indir_ref:
        opt.use_renv = True

    if len(opt.resume_mlps) == 0:
        opt.resume_mlps = opt.frozen_mlps

    if len(opt.relsdf_loss_weight_sched) == 4:
        opt.relsdf_loss_weight_sched.insert(2, opt.relsdf_loss_start_iter)

    if len(opt.backsdf_loss_weight_sched) == 4:
        opt.backsdf_loss_weight_sched.insert(2, opt.backsdf_loss_start_iter)

    if len(opt.cauchy_loss_weight_sched) == 4:
        opt.cauchy_loss_weight_sched.insert(2, opt.cauchy_loss_start_iter)

    if len(opt.eikonal_loss_weight_sched) == 4:
        opt.eikonal_loss_weight_sched.insert(2, opt.eikonal_loss_start_iter)
    
    if len(opt.orientation_loss_weight_sched) == 4:
        opt.orientation_loss_weight_sched.insert(2, opt.orientation_loss_start_iter)

    if opt.unwrap_env_sphere:
        opt.test = True

    if len(opt.test_ids) == 1 and opt.test_ids[0] == -1:
        opt.test_ids = []
        
    return opt