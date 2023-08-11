import torch

from nerf.options import config_parser
from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.utils import *

from functools import partial
from loss import huber_loss

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    
    opt = config_parser()

    # if opt.debug:
    #     assert opt.test, "debug mode only works in test mode"

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
    
    if opt.patch_size > 1:
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."


    if opt.ff:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --ff"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    print(opt)
    
    seed_everything(opt.seed)

    env_opt = None
    if opt.env_sph_mode or opt.render_env_on_sphere:
        env_dataset_config = opt.env_dataset_config.strip()
        if opt.sph_renderer == 'filament':
            from nerf.sph_loader import EnvDataset
            from nerf.sph_loader import config_parser
        # elif opt.sph_renderer == 'mitsuba':
        #     from nerf.sph_loader_mi import EnvDataset
        #     from nerf.sph_loader_mi import config_parser
        
        env_opt = config_parser(env_dataset_config)

    model = NeRFNetwork(
        encoding="hashgrid",
        encoding_dir=opt.encoding_dir,
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        use_sdf = opt.use_sdf,
        hidden_dim=opt.hidden_dim,
        num_layers=opt.num_layers,
        num_layers_color=opt.num_layers_color,
        hidden_dim_color=opt.hidden_dim_color,
        num_layers_bg=opt.num_layers_bg,
        num_levels=opt.num_levels,
        geo_feat_dim=opt.geo_feat_dim,
        opt=opt,
        env_opt=env_opt
    )
    
    print(model)

    if opt.color_l1_loss:
        # use L1 loss for color_l
        criterion = torch.nn.L1Loss(reduction='none')
    else:
        criterion = torch.nn.MSELoss(reduction='none')
    #criterion = partial(huber_loss, reduction='none')
    #criterion = torch.nn.HuberLoss(reduction='none', beta=0.1) # only available after torch 1.10 ?

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.unwrap_env_sphere:
        from nerf.render_func import unwrap_env_sphere
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, use_checkpoint=opt.ckpt)
        material = {
            "roughness": opt.unwrap_roughness, # TODO make it opt
            "metallic": 1.0,
            "color": [opt.unwrap_color_intensity for i in range(3)] #[0.7, 0.7, 0.7]
        }
        opt.env_sph_radius = opt.env_sph_radius * opt.scale
        print(f"opt.unwrap_env_id={opt.unwrap_env_id}")
        unwrap_env_sphere(trainer, device, material=material, env_net_index=opt.unwrap_env_id, use_specular_color=True)
        # for i in range(11):
        #     unwrap_env_sphere(trainer, device, material=material, env_net_index=i, use_specular_color=True)
        print("unwrap done")
        exit()   

    if opt.test:
        
        metrics = [PSNRMeter(),] # LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)
                
        if opt.cuda_ray and opt.extra_state_full_update:
            with torch.cuda.amp.autocast(enabled=opt.fp16):
                model.reset_extra_state()
                model.update_extra_state(full_update=True)
                for i in range(16):
                    model.update_extra_state(full_update=True)
                    

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            if opt.debug:
                test_id = [opt.debug_id]
            else:
                test_id = opt.test_ids if len(opt.test_ids) > 0 else None
            if opt.env_sph_mode or opt.render_env_on_sphere:
                test_loader = EnvDataset(env_opt, device=device, type=opt.test_split, opt=opt).dataloader(test_ids=test_id)
                opt.env_sph_radius = test_loader._data.sph_radius
            else:
                test_loader = NeRFDataset(opt, device=device, type=opt.test_split).dataloader(test_ids=test_id)

            cfg_train_opt(opt, trainer.epoch)
            if opt.dir_only:
                opt.indir_ref = False
            
            if test_loader.has_gt:
                # TODO: env_rot_degree_range
                trainer.evaluate(test_loader, None, opt.env_rot_degree_range) # blender has gt, so evaluate it.
    
            # trainer.test(test_loader, write_video=True) # test and save video
            
            # trainer.save_mesh(resolution=256, threshold=10)
    
    else:

        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr, opt.plr, opt.slr, opt.elr), betas=(0.9, 0.99), eps=1e-15)

        if opt.env_sph_mode:
            train_loader = EnvDataset(env_opt, opt=opt, device=device, type='train').dataloader()
            opt.env_sph_radius = train_loader._data.sph_radius
        else:
            train_loader = NeRFDataset(opt, device=device, type='train').dataloader(batch_size=opt.image_batch)
        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        metrics = [PSNRMeter(),] #, LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, \
                    ema_decay=0.95 if not opt.geometric_init else None,
                    fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, \
                    use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, max_keep_ckpt=opt.max_keep_ckpt)
        
        # cfg_train_opt(opt, trainer.epoch)

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()
        
        else:

            if opt.debug:
                test_id = [opt.debug_id]
            else:
                test_id = opt.test_ids if len(opt.test_ids) > 0 else None
            if opt.env_sph_mode:
                valid_loader = EnvDataset(env_opt, opt=opt, device=device, type='val', downscale=1).dataloader(test_ids=test_id, test_skip=opt.test_skip)
            else:
                valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader(test_ids=test_id, test_skip=opt.test_skip)              

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            trainer.train(train_loader, valid_loader, max_epoch)

            # also test
            # test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
            
            # if test_loader.has_gt:
            #     trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            
            # trainer.test(test_loader, write_video=True) # test and save video

            if opt.env_sph_mode and not opt.train_renv:
                from nerf.sph_loader import extract_env_ckpt
                name = f'{trainer.name}_ep{trainer.epoch:04d}'
                file_path = f"{trainer.ckpt_path}/{name}.pth"
                extract_env_ckpt(file_path)

            if not opt.env_sph_mode:
                threshold = 10
                if trainer.opt.use_sdf:
                    threshold = 0
                trainer.save_mesh(resolution=256, threshold=threshold)