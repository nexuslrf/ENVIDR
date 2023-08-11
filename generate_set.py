import configargparse
from nerf.sph_loader import trans_t, rot_phi, rot_theta, pose_spherical

import open3d as o3d
import open3d.visualization.rendering as rendering
import imageio
import json
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def config_parser(env_dataset_config):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--camera_angle_x", type=float, default=0.6194058656692505, help="camera angle x")
    parser.add_argument("--env_images_base_dir", type=str, help="ktx images base dir")
    parser.add_argument("--env_images_list", type=str, help="ktx images list")
    parser.add_argument("--mesh_path", type=str, help="mesh path")
    parser.add_argument("--render_image_H", type=int, default=800, help="render image H")
    parser.add_argument("--render_image_W", type=int, default=800, help="render image W")
    parser.add_argument("--radius", type=float, default=4.0, help="radius")
    parser.add_argument("--num_metalness", type=int, default=5, help="num metalness")
    parser.add_argument("--num_roughness", type=int, default=5, help="num roughness")
    parser.add_argument("--workspace", type=str, help="workspace")
    parser.add_argument("--type", type=str, help="type")

    env_opt = parser.parse_args(f"--config {env_dataset_config}")
    return env_opt

opt = config_parser("./configs/val_config.ini")
# init local render
local_render = rendering.OffscreenRenderer(800, 800)
mesh = o3d.io.read_triangle_model(opt.mesh_path)
local_render.scene.add_model("mesh", mesh)
local_render.scene.scene.enable_sun_light(False)            

theta_phi_min = [0, -90]
theta_phi_max = [360, 90]

output_json = {
    "camera_angle_x": opt.camera_angle_x,
    "frames": []
}


os.makedirs(os.path.join(opt.workspace, opt.type), exist_ok=True)

min_rough = 0.001
max_rough = 1.0
min_metal = 0.1
max_metal = 1.0
roughnesses = torch.arange(min_rough, max_rough, (max_rough - min_rough) / opt.num_roughness)
# metalnesses = torch.arange(min_metal, max_metal, (max_metal - min_metal) / opt.num_metalness)
# try to collect balls with stronger reflectance
# roughnesses /= roughnesses.max()
# metalnesses /= metalnesses.max()
num_close_to_0_metal = int(opt.num_metalness * 0.4)
num_close_to_1_metal = opt.num_metalness - num_close_to_0_metal
metalnesses_close_to_0 = abs(torch.normal(0, 0.25, size=(num_close_to_0_metal, )))
metalnesses_close_to_1 =  1 - abs(torch.normal(0, 0.25, size=(num_close_to_1_metal, )))
# print(f"metalnesses_close_to_0={metalnesses_close_to_0}")
# print(f"metalnesses_close_to_1={metalnesses_close_to_1}")
metalnesses = torch.cat([metalnesses_close_to_0, metalnesses_close_to_1])
# print("roughnesses=", roughnesses)
print("metalnesses=", metalnesses)
# metalnesses  = torch.exp(metalnesses)
# metalnesses /= metalnesses.max()
roughnesses = torch.pow(roughnesses, 2)
print("roughnesses=", roughnesses)
# print("metalnesses=", metalnesses)

env_images_names = []
with open(opt.env_images_list, "r") as f:
    for line in f.readlines():
        env_images_names.append(line.strip())
# print("env_images_names=", env_images_names)

material_params = {
    "reflectance": 0.5,
    "clearcoat": 0.0,
    "clearcoat_roughness": 0.0,
    "anisotropy": 0.0
}
material = rendering.MaterialRecord()
for key, val in material_params.items():
    setattr(material, "base_" + key, val)

class RenderWrapper():
    def __init__(self, opt) -> None:
        self.W = opt.render_image_W
        self.H = opt.render_image_H
        self.radius = opt.radius
        self.focal = self.W / (2 * np.tan(opt.camera_angle_x / 2))
        self.camera_intrinsics = np.array([
                        [self.focal, 0, self.W/2],
                        [0, self.focal, self.H/2],
                        [0, 0, 1]
                    ])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.y2z = np.array([[1,0,0,0],
                        [0,0,1,0],
                        [0,-1,0,0],
                        [0,0,0,1]])

    def get_transform_matrix_extrinsics(self, theta, phi):
        transform_matrix = pose_spherical(theta, phi, self.radius)
        extrinsics = transform_matrix @ self.blender2opencv
        extrinsics = np.linalg.inv(self.y2z @ extrinsics)
        return transform_matrix, extrinsics

    def render_image(self, theta, phi, env_path, render=None):
        transform_matrix, extrinsics = self.get_transform_matrix_extrinsics(theta, phi)
        if render is None:
            render = self.render
        assert render is not None

        render.setup_camera(self.camera_intrinsics, extrinsics, self.H, self.W)
        render.scene.scene.set_indirect_light(env_path)
        render.scene.scene.set_indirect_light_intensity(60000)

        img = render.render_to_image()
        img = np.asarray(img)
        depth = render.render_to_depth_image()
        depth = np.asarray(depth)
        alpha = 1-(depth ==1)
        alpha = (alpha[...,None] * 255).astype(np.uint8)
        img = np.concatenate([img, alpha], axis=-1)      

        return transform_matrix, img          

i = 0

fig, axs = plt.subplots(opt.num_roughness, opt.num_metalness, figsize=(10,10))
fig.tight_layout(pad=1)

METALLIC_THRESHOLD = 0.5

renderer_wrapper = RenderWrapper(opt)
progress_bar = tqdm(total=len(env_images_names)*len(roughnesses)*len(metalnesses))
for env_image_name in env_images_names:
    for rough_index, roughness in enumerate(roughnesses):
        for metal_index, metallic in enumerate(metalnesses):
            # https://google.github.io/filament/Filament.html#toc4.8
            
            if metallic >= METALLIC_THRESHOLD:
                # metal base color 170-255 sRGB
                base_color = torch.randint(170, 255, size=(3, ))
            else:
                # non-metal base color 50-240 sRGB (strict range) 
                base_color = torch.randint(50, 240, size=(3, ))
            # print(f"base_color={base_color}")
            base_color = base_color.float() / 255.0
            base_color = np.array([base_color[0], base_color[1], base_color[2], 1.0])
            material.base_color = base_color
            setattr(material, "base_" + "roughness", roughness)
            setattr(material, "base_" + "metallic", metallic)
            material.shader = "defaultLit"
            local_render.scene.update_material(material)
            env_images_path = f"{opt.env_images_base_dir}/{env_image_name}/{env_image_name}"  
            theta = np.random.uniform(theta_phi_min[0], theta_phi_max[0]) # 180 # for golf
            phi = np.random.uniform(theta_phi_min[1], theta_phi_max[1]) # 0 # for golf
            transform_matrix, img = renderer_wrapper.render_image(theta, phi, env_images_path, local_render) 
            
            # normal image
            material.shader = "normals"
            local_render.scene.update_material(material)
            normal_img = local_render.render_to_image()

            dataset_image_path = f"{opt.type}/r_{i}.png"
            # print(f"roughness={roughness}, metallic={metallic}, dataset_image_path={dataset_image_path}, theta={theta}, phi={phi}")
            imageio.imwrite(f"{opt.workspace}/{dataset_image_path}", img)
            imageio.imwrite(f"{opt.workspace}/{opt.type}/r_{i}_normal.png", normal_img)
            axs[rough_index, metal_index].imshow(img)
            axs[rough_index, metal_index].set_title(f'[R={roughness:.2f}, M={metallic:.2f}]', size=8)
            axs[rough_index, metal_index].axis('off')
            output_json["frames"].append({
                "file_path": dataset_image_path,
                "transform_matrix": transform_matrix.tolist(),
                "env_image_name": env_image_name,
                "roughness": roughness.item(),
                "metallic": metallic.item(),
                "color": base_color.tolist()
            })
            i += 1        
            progress_bar.update(1)

    full_figure_path = os.path.join(opt.workspace, f"full_figure_{env_image_name}_{opt.type}.png")
    fig.savefig(full_figure_path)
    print(f"outputed {full_figure_path}")

with open(os.path.join(opt.workspace, f"transforms_{opt.type}.json"), "w") as f:
    json.dump(output_json, f, indent=4)
# test set same as val set
with open(os.path.join(opt.workspace, f"transforms_test.json"), "w") as f:
    json.dump(output_json, f, indent=4)

del local_render