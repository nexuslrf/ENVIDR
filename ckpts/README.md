# Checkpoints for Pre-trained Rendering MLPs

The ckpts here contains the mininal weight parameters from our pre-trained rendering models that are needed for our experiments.
These weights are exactly the same as the ones shown in [`../demo.ipynb`](../demo.ipynb) and [`../demo`](../demo/), but in a slight different store format.
The full model checkpoints can be accessed through the [Google Drive link](https://drive.google.com/drive/folders/1eyNyUhdp6S4ThqxwRzTYQZAL9xTY01yN?usp=sharing).

- `rendering_mlps.pth` contains all the weights for the rendering MLPs (FP32, 72 KB).
- `env_ckpts/` contains the weights for the environment MLPs, the suffix id indicates the environment id (FP32, 236KB each).