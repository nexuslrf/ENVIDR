# *ENVIDR*: Implicit Differentiable Render with Neural Environment Lighting

The official PyTorch codebase for ICCV'23 paper "*ENVIDR*: Implicit Differentiable Render with Neural Environment Lighting"

| [Project Page](http://nexuslrf.github.io/ENVIDR) | [Paper](https://arxiv.org/abs/2303.13022) |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nexuslrf/ENVIDR/blob/main/demo.ipynb)

![img](figs/renderer_overview.jpg)

## Updates

[2023/09/03] Upload the pretrained checkpoints for rendering MLPs at [`./ckpts`](./ckpts/).
[2023/08/11] Release more training and evaluation codes. The pre-trained checkpoints will be updated later.

## Setup

The our code is mainly based on the awesome third-party [torch-ngp implementation](https://github.com/ashawkey/torch-ngp). The instructions to set up the running environment are as follows:

**Clone Repo:**

```bash
git clone --recursive git@github.com:nexuslrf/ENVIDR.git
cd ENVIDR
```

**Install Packages:**

```bash
conda create -n envidr python=3.8
conda activate envidr
# you might need to manually install torch>=1.10.0 based on your device
pip install -r requirements.txt
```

**Torch Extensions:**

By default, `cpp_extension.load` will build the extension at runtime. However, this may be inconvenient sometimes.

We *recommend* pre-build all extensions by running the following script:

```bash
# install all extension modules locally
bash build_ext.sh
```

## Dataset Preparation

### General Scenes/Objects

We use the format of the original [nerf-synthetic](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi?usp=drive_link) dataset for the model training. The download links for dataset are shown below:

* Original [nerf-synthetic](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi?usp=drive_link)
* RefNeRF's [shiny-synthetic](https://storage.googleapis.com/gresearch/refraw360/ref.zip)

We by default put the datasets under the folder `data/`

### Spheres for Pre-trained Rendering MLPs

To learn the pre-trained rendering MLPs, you additionally need download a set of HDRI environment light images. In our experiments, we use the 11 [HDRIs](https://github.com/google/filament/tree/main/third_party/environments) provided by [Filament](https://github.com/google/filament) renderer. We provide a script to download and convert HDRI for Filament renderer:

```bash
bash prepare_hdri.sh
```

In case you fail to convert the HDRI into KTX file, we also provide our [pre-computed KTX file](https://drive.google.com/file/d/1Khqt1g244HkVFnIxFNLMnupa9_V7-s75/view?usp=sharing).

After obtaining the converted KTX files for environment map, you can run `generate_set.py` to verify the rendering results and also a get a set of sample images for the evaluation purpose during training.

```bash
python generate_set.py
```

The results will be written to `data/env_sphere/env_dataset` by default.

## Training

### Neural Renderer

*You can use our provided [pre-trained rendering MLPs](./ckpts/) to skip this step.*

To train a neural renderer from scratch:

```bash
python main_nerf.py --config ./configs/neural_renderer.ini
# Optionally to train rendering MLPs for indirect reflection.
python main_nerf.py --config ./configs/neural_renderer_renv.ini
```

The results by default will be saved at `exps/`

We also provide the checkpoints of our pre-trained neural renderer.

### General Scenes

```bash
python main_nerf.py --config ./configs/scenes/toaster.ini
```

You can get decent results after 500 epochs of the training.

Note the following flags in `.ini` file is for enabling the interreflections:

```ini
use_renv = True # color encoding MLP $E_{ref}$ in Eqn. 13 of the paper
; indir_ref = True
indir_ref_start_iter = 140 # indir_ref is enabled after 140 epoches
learn_indir_blend = True
grad_rays_start_iter=100
grad_rays_scale=0.05

; dir_only = True # only render direct illumination
; indir_only = True # only render indirect illumination
```

BTW, these pre-trained rendering MLPs can also be used with NeuS-like (without hash encoding) geometry models:

```bash
python main_nerf.py --config ./configs/scenes/materials_neus.ini
```

## Applications

### Extract Environment Map

```bash
python main_nerf.py --config configs/unwrap_scene.ini \
    --workspace exps/scenes/toaster --swap_env_path exps/scenes/toaster/checkpoints/ngp_ep0500.pth \
    --unwrap_color_intensity=1.0 --unwrap_roughness=0.4
```

Note you need to manually tune `unwrap_roughness` to get a clear & detailed environment map.

### Relight the Scene

```bash
python main_nerf.py --config configs/scenes/toaster.ini --test \
    --swap_env_path exps/envs_all_11_unit_en/checkpoints/env_ckpts/env_net_3.pth \
    --sh_degree 4 --hidden_dim_env 160 \
    --val_folder_name relighting \
    --intensity_scale=0.8 --roughness_scale=0.8
```

Note you can manually tune `intensity_scale` and `roughness_scale` to get the desired relighting results.

### Rotate the Environment

```bash
python main_nerf.py --config configs/scenes/toaster.ini --test \
    --test_ids 57 --val_folder_name rot_env \
    --env_rot_degree_range 0 360 5 # [degree_start, degree_end, num_views]
```

## Acknowledgement

We also used the following awesome codebases to implement this project:

* Filament: https://google.github.io/filament/Filament.html
* instant-ngp: https://github.com/NVlabs/instant-ngp
* differentiable hash encoder: https://github.com/autonomousvision/monosdf
* ide encoder: https://github.com/google-research/multinerf/blob/main/internal/ref_utils.py

## Citation

```
@article{liang2023envidr,
  title={ENVIDR: Implicit Differentiable Renderer with Neural Environment Lighting},
  author={Liang, Ruofan and Chen, Huiting and Li, Chunlin and Chen, Fan and Panneer, Selvakumar and Vijaykumar, Nandita},
  journal={arXiv preprint arXiv:2303.13022},
  year={2023}
}
```
