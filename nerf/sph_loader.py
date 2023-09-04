from collections import OrderedDict
import numpy as np
import configargparse

import open3d as o3d
import open3d.visualization.rendering as rendering
import imageio
from torch.utils.data import DataLoader

import os
import json
import torch
import tqdm

from .provider import nerf_matrix_to_ngp
from .utils import get_rays, trans_t, rot_theta, rot_phi

def config_parser(env_dataset_config):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--camera_angle_x", type=float, default=0.6194058656692505, help="camera angle x")
    parser.add_argument("--sph_radius", type=float, default=0.95, help="sph radius")
    parser.add_argument("--env_images_base_dir", type=str, help="env images base dir, env should in ktx format (Filament)")
    parser.add_argument("--env_images_list", type=str, help="env images list")
    parser.add_argument("--mesh_path", type=str, help="mesh path")
    parser.add_argument("--render_image_H", type=int, default=800, help="render image H")
    parser.add_argument("--render_image_W", type=int, default=800, help="render image W")
    parser.add_argument("--radius", type=float, default=4.0, help="radius")
    parser.add_argument("--num_train_images", type=int, default=100, help="num train images")
    parser.add_argument("--val_root_path", type=str, help="val root path")
    parser.add_argument("--test_root_path", type=str, help="test root path")
    parser.add_argument("--num_workers", type=int, default=4, help="train loader num workers")
    parser.add_argument("--vary_roughness", action="store_true", help="vary roughness")
    parser.add_argument("--vary_metallic", action="store_true", help="vary metallic")
    parser.add_argument("--vary_base_color", action="store_true", help="vary base colr")
    parser.add_argument("--rendering_cuda_device", type=str, default="", help="rendering cuda device")

    if env_dataset_config != '':
        env_opt = parser.parse_args(f"--config {env_dataset_config}")
    else:
        env_opt = parser.parse_args("")
    with open(env_opt.env_images_list, "r") as f:
        env_opt.env_images_names =  f.read().splitlines()
    env_opt.env_image_name_to_index = {}
    for index, name in enumerate(env_opt.env_images_names):
        env_opt.env_image_name_to_index[name] = index
    return env_opt

DEFAULT_MATERIAL = {
    "reflectance": 0.5,
    "clearcoat": 0.0,
    "clearcoat_roughness": 0.0,
    "anisotropy": 0.0
}
METALLIC_THRESHOLD = 0.5

R_MATERIAL = {
    "roughness": 0.0,
    "metallic": 1.0,
    "reflectance": 0.5,
    "clearcoat": 0.0,
    "clearcoat_roughness": 0.0,
    "anisotropy": 0.0,
    "color": np.array([0.8, 0.8, 0.8, 1.0])
}

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array(
                [[-1,0,0,0],
                [ 0,0,1,0],
                [ 0,1,0,0],
                [ 0,0,0,1]]) @ c2w
    return c2w

class EnvDataset:
    def __init__(self, env_opt, type, device, opt, downscale=1.0):
        # TODO: downscale
        # TODO: preload to GPU
        if env_opt.rendering_cuda_device != '':
            os.environ['CUDA_VISIBLE_DEVICES'] = env_opt.rendering_cuda_device
        self.env_opt = env_opt
        self.opt = opt
        self.type = type # train, val, test
        self.device = device
        self.camera_angle_x = self.env_opt.camera_angle_x
        self.env_images_base_dir = self.env_opt.env_images_base_dir # used for set_indirect_light (train)
        self.env_images_names =  self.env_opt.env_images_names # val / test
        self.mesh_path = self.env_opt.mesh_path
        self.H = self.env_opt.render_image_H
        self.W = self.env_opt.render_image_W
        self.radius = self.env_opt.radius
        self.sph_radius = self.env_opt.sph_radius * self.opt.scale
        self.num_train_images = self.env_opt.num_train_images
        self.val_root_path = self.env_opt.val_root_path
        self.test_root_path = self.env_opt.test_root_path
        self.focal = self.W / (2 * np.tan(self.camera_angle_x / 2))
        self.camera_intrinsics = np.array([
                [self.focal, 0, self.W/2],
                [0, self.focal, self.H/2],
                [0, 0, 1]
            ])

        fl_x, fl_y = self.focal, self.focal
        cx, cy = self.W/2, self.H/2
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.y2z = np.array([[1,0,0,0],
                        [0,0,1,0],
                        [0,-1,0,0],
                        [0,0,0,1]])

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if  self.training else -1

        assert opt.workspace is not None
        os.makedirs(opt.workspace, exist_ok=True)

        self.render = None
        self.theta_phi_min = [0, -90]
        self.theta_phi_max = [360, 90]
        self.num_envs = len(self.env_images_names)
        
        self.rendered_images, self._poses, self.env_net_indices = None, None, None
        if self.type in ("val", "test"):
            # load from disk
            self.rendered_images, self._poses, self.env_net_indices, self.materials, self._nerf_poses = [], [], [], [], []
            if self.opt.train_renv:
                self._nerf_poses = []
                self.json_env_images_names= []
            json_path = os.path.join(self.val_root_path if self.type == "val" else self.test_root_path, f"transforms_{self.type}.json")
            with open(json_path, "r") as f:
                json_file = json.load(f)
                for frame in json_file["frames"]:
                    _pose = np.array(frame["transform_matrix"])
                    if self.opt.train_renv:
                        self._nerf_poses.append(_pose)
                    _pose = nerf_matrix_to_ngp(_pose, scale=self.opt.scale, offset=self.opt.offset)
                    self._poses.append(_pose)
                    _image = imageio.imread(os.path.join(self.val_root_path if self.type == "val" else self.test_root_path, frame["file_path"]))
                    self.rendered_images.append(_image)
                    env_image_name = frame["env_image_name"]
                    assert env_image_name in self.env_opt.env_image_name_to_index
                    if self.opt.train_renv:
                        self.json_env_images_names.append(env_image_name)
                    env_net_index = self.env_opt.env_image_name_to_index[env_image_name]
                    self.env_net_indices.append(env_net_index)
                    material = {
                        "roughness": frame['roughness'],
                        "metallic": frame['metallic'],
                        "color": frame['color']
                    }
                    self.materials.append(material)
            self._poses = torch.from_numpy(np.stack(self._poses, axis=0))
            self.rendered_images = torch.from_numpy(np.stack(self.rendered_images, axis=0) / 255.)
            self.env_net_indices = torch.from_numpy(np.stack(self.env_net_indices, axis=0)).int()
            # self.materials = torch.from_numpy(np.stack(self.materials, axis=0)) # Cannot do this since self.material is dictionary
            if self.type == "val":
                self.num_val_images = len(self.rendered_images)
            elif self.type == "test":
                self.num_test_images = len(self.rendered_images)
        self.num_images = self.num_train_images if self.type == "train" else self.num_val_images if self.type == "val" else self.num_test_images
        self.error_map = None
        # exit()

    def get_transform_matrix_extrinsics(self, theta, phi):
        transform_matrix = pose_spherical(theta, phi, self.radius)
        extrinsics = transform_matrix @ self.blender2opencv
        extrinsics = np.linalg.inv(self.y2z @ extrinsics)
        return transform_matrix, extrinsics

    def render_image_w_extrinsics(self, extrinsics, env_path=None, material_dict={}, render=None, with_alpha=False):
        if render is None:
            render = self.render
        assert render is not None

        if hasattr(self, "material"):
            material = self.material
        else:
            material = render.scene.materials[0]

        render.setup_camera(self.camera_intrinsics, extrinsics, self.H, self.W)
        
        if env_path is not None:
            render.scene.scene.set_indirect_light(env_path)
            render.scene.scene.set_indirect_light_intensity(60000)

        if material_dict:
            for k, v in material_dict.items():
                setattr(material, f'base_{k}', v)
            render.scene.update_material(material)
        
        img = render.render_to_image()
        img = np.asarray(img)
        # alpha is not necessary for training
        if with_alpha:
            depth = render.render_to_depth_image()
            depth = np.asarray(depth)
            alpha = 1-(depth ==1)
            alpha = (alpha[...,None] * 255).astype(np.uint8)
            img = np.concatenate([img, alpha], axis=-1)     
        return img

    def render_image(self, theta, phi, env_path=None, material_dict={}, render=None, with_alpha=False):

        transform_matrix, extrinsics = self.get_transform_matrix_extrinsics(theta, phi)
        img = self.render_image_w_extrinsics(extrinsics, env_path=env_path, material_dict=material_dict, render=render, with_alpha=with_alpha)

        return transform_matrix, img  

    def collate(self, index):
        if self.render is None and (self.type == "train" or self.opt.train_renv):
            self.render = rendering.OffscreenRenderer(800, 800)
            mesh = o3d.io.read_triangle_model(self.mesh_path)
            self.render.scene.add_model("mesh", mesh)
            env_image_name = self.env_images_names[0]
            env_images_path = f"{self.env_images_base_dir}/{env_image_name}/{env_image_name}"
            self.render.scene.scene.set_indirect_light(env_images_path)
            self.render.scene.scene.set_indirect_light_intensity(60000)
            self.render.scene.scene.enable_sun_light(False)  

            self.material = rendering.MaterialRecord()
            self.material.shader = "defaultLit"
            for key, val in DEFAULT_MATERIAL.items():
                setattr(self.material, "base_" + key, val)
            
            self.render.scene.update_material(self.material)

        B = len(index) # a list of length 1

        if self.type in ("val", "test"):
            images = self.rendered_images[index] #.to(self.device)
            poses = self.poses[index] #.to(self.device) # [B, 4, 4]
            env_net_indices = self.env_net_indices[index]
            assert B == 1
            material = self.materials[index[0]]
            if self.opt.train_renv:
                transform_matrix = self._nerf_poses[index[0]]
                extrinsics = transform_matrix @ self.blender2opencv
                extrinsics = np.linalg.inv(self.y2z @ extrinsics)
                env_image_name = self.json_env_images_names[index[0]]
                env_images_path = f"{self.env_images_base_dir}/{env_image_name}/{env_image_name}"
                r_images = self.render_image_w_extrinsics(extrinsics, env_path=env_images_path, material_dict=R_MATERIAL)
                r_images = torch.from_numpy(np.stack(r_images, axis=0))
                r_images = r_images / 255.0
        else:
            poses = []
            images = []
            env_net_indices = []
            r_images = []
            for i in index:
                theta, phi = np.random.uniform(low=0, high=360), np.random.uniform(low=-90, high=90)
                env_images_path = None
                if self.num_train_images > 1:
                    env_image_name = np.random.choice(self.env_images_names, size=1)[0]
                    # TODO: use multiple renders for different ktx images instead
                    # you need to verify. if the speed won't change much, then it's fine
                    env_images_path = f"{self.env_images_base_dir}/{env_image_name}/{env_image_name}"

                material = {}
                # TODO: random sampler might need to be changed
                if self.env_opt.vary_roughness:
                    roughness = np.random.uniform(low=0, high=1 if not self.opt.train_renv else 0.6)
                    material["roughness"] = roughness ** 2 # low roughness is more challenging
                if self.env_opt.vary_metallic:
                    if np.random.uniform(low=0, high=1) < 0.5:
                        # 50% non metallic (close to 0)
                        material["metallic"] = abs(torch.normal(0, 0.25, size=(1, ))[0])
                    else:
                        # 50% metallic (close to 1)
                        material["metallic"] = 1 - abs(torch.normal(0, 0.25, size=(1, ))[0])
                if self.env_opt.vary_base_color:
                    if material["metallic"] >= METALLIC_THRESHOLD:
                        # metal base color 170-255 sRGB
                        base_color = torch.randint(170, 255, size=(3, ))
                    else:
                        # non-metal base color 50-240 sRGB (strict range) 
                        base_color = torch.randint(50, 240, size=(3, ))                    
                    material["color"] = np.append(base_color.float() / 255.0, [1.0])
                
                transform_matrix, image = self.render_image(theta, phi, env_images_path, material)

                if self.opt.train_renv:
                    _, r_image = self.render_image(theta, phi, env_images_path, R_MATERIAL)
                    r_images.append(r_image)

                pose = nerf_matrix_to_ngp(transform_matrix, scale=self.opt.scale, offset=self.opt.offset)
                poses.append(pose)
                images.append(image)
                env_net_indices.append(self.env_opt.env_image_name_to_index[env_image_name])
            poses = torch.from_numpy(np.stack(poses, axis=0)) #.to(self.device) # [B, 4, 4]
            images = torch.from_numpy(np.stack(images, axis=0)) #.to(self.device) # [B, H, W, 4]
            env_net_indices = torch.from_numpy(np.stack(env_net_indices, axis=0)) # [B, H, W, 4]
            images = images / 255.0
            if self.opt.train_renv:
                r_images = torch.from_numpy(np.stack(r_images, axis=0))
                r_images = r_images / 255.0
        
        # rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, self.error_map, self.opt.patch_size)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': None, #rays['rays_o'].clone(),
            'rays_d': None, #rays['rays_d'].clone(),
            'poses': poses.to(torch.float),
            'images': images.to(torch.float),
            'env_net_indices': env_net_indices,
            'material': material, # TODO
        }
        if self.opt.train_renv:
            results["r_images"] = r_images

        return results

    # def worker_init_fn(self, worker_id):
    #     self.render = rendering.OffscreenRenderer(800, 800)
    #     mesh = o3d.io.read_triangle_model(self.mesh_path)
    #     self.render.scene.add_model("mesh", mesh)

    @property
    def poses(self):
        if self._poses is not None:
            poses = self._poses
        else:
            poses = []
            theta_phi_combs = np.random.uniform(low=self.theta_phi_min, high=self.theta_phi_max, size=(self.num_images, 2))
            for theta, phi in theta_phi_combs:
                transform_matrix = pose_spherical(theta, phi, self.radius)
                poses.append(nerf_matrix_to_ngp(transform_matrix, scale=self.opt.scale, offset=self.opt.offset))
            poses = torch.from_numpy(np.stack(poses, axis=0)) # .to(self.device)
        # print("poses.shape=", poses.shape)
        return poses

    def dataloader(self, test_ids=None, test_skip=1):

        size = self.num_images
        if test_ids is None:
            test_ids = list(range(0,size,test_skip))
        num_workers = 0 
        if self.type == "train":
            num_workers = self.env_opt.num_workers
        # loader = DataLoader(test_ids, batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=16, pin_memory=True, prefetch_factor=8, pin_memory_device=str(self.device), persistent_workers=True)
        if num_workers > 0:
            # loader = DataLoader(test_ids, batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=num_workers, pin_memory=True, prefetch_factor=8, pin_memory_device=str(self.device), persistent_workers=True)
            loader = DataLoader(test_ids, batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=num_workers, pin_memory=True, prefetch_factor=8, persistent_workers=True)
        else:
            loader = DataLoader(test_ids, batch_size=1, collate_fn=self.collate, shuffle=self.training)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = True
        return loader
    
def extract_env_ckpt(ckpt_path):
    save_dict = torch.load(ckpt_path)
    model_dict = save_dict['model']
    
    parent_dir = os.path.dirname(ckpt_path)
    os.makedirs(f'{parent_dir}/env_ckpts', exist_ok=True)
    env_id = 0
    found_id = False
    pbar = tqdm.tqdm()
    pbar.write(f'Extracting env ckpts from {ckpt_path} to {parent_dir}/env_ckpts')
    while found_id or env_id == 0:
        env_net_dict = {}
        found_id = False
        for k, v in model_dict.items():
            prefix = f'env_nets.{env_id}.'
            if k.startswith(prefix):
                env_net_dict['env_net' + k[len(prefix):]] = v
                found_id = True
        if found_id:
            env_dict = OrderedDict()
            env_dict['model'] = env_net_dict
            torch.save(env_dict, f'{parent_dir}/env_ckpts/env_net_{env_id}.pth')
            pbar.update()
            env_id += 1