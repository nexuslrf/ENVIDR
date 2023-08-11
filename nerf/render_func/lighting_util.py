import numpy as np
from PIL import Image
import os
import torch
from torchvision.utils import make_grid
from os.path import join
import torch.nn.functional as F
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# from .rot import _warn_degree
def _warn_degree(angles):
    if (np.abs(angles) > 2 * np.pi).any():
        print((
            "Some input value falls outside [-2pi, 2pi]. You sure inputs are "
            "in radians"))

def uniform_sample_sph(n, r=1, convention='lat-lng'):
    r"""Uniformly samples points on the sphere
    [`source <https://mathworld.wolfram.com/SpherePointPicking.html>`_].

    Args:
        n (int): Total number of points to sample. Must be a square number.
        r (float, optional): Radius of the sphere. Defaults to :math:`1`.
        convention (str, optional): Convention for spherical coordinates.
            See :func:`cart2sph` for conventions.

    Returns:
        numpy.ndarray: Spherical coordinates :math:`(r, \theta_1, \theta_2)`
        in radians. The points are ordered such that all azimuths are looped
        through first at each elevation.
    """
    n_ = np.sqrt(n)
    if n_ != int(n_):
        raise ValueError("%d is not perfect square" % n)
    n_ = int(n_)

    pts_r_theta_phi = []
    for u in np.linspace(0, 1, n_):
        for v in np.linspace(0, 1, n_):
            theta = np.arccos(2 * u - 1) # [0, pi]
            phi = 2 * np.pi * v # [0, 2pi]
            pts_r_theta_phi.append((r, theta, phi))
    pts_r_theta_phi = np.vstack(pts_r_theta_phi)

    # Select output convention
    if convention == 'lat-lng':
        pts_sph = _convert_sph_conventions(
            pts_r_theta_phi, 'theta-phi_to_lat-lng')
    elif convention == 'theta-phi':
        pts_sph = pts_r_theta_phi
    else:
        raise NotImplementedError(convention)

    return pts_sph


def cart2sph(pts_cart, convention='lat-lng'):
    r"""Converts 3D Cartesian coordinates to spherical coordinates.

    Args:
        pts_cart (array_like): Cartesian :math:`x`, :math:`y` and
            :math:`z`. Of shape N-by-3 or length 3 if just one point.
        convention (str, optional): Convention for spherical coordinates:
            ``'lat-lng'`` or ``'theta-phi'``:

            .. code-block:: none

                   lat-lng
                                            ^ z (lat = 90)
                                            |
                                            |
                       (lng = -90) ---------+---------> y (lng = 90)
                                          ,'|
                                        ,'  |
                   (lat = 0, lng = 0) x     | (lat = -90)

            .. code-block:: none

                theta-phi
                                            ^ z (theta = 0)
                                            |
                                            |
                       (phi = 270) ---------+---------> y (phi = 90)
                                          ,'|
                                        ,'  |
                (theta = 90, phi = 0) x     | (theta = 180)

    Returns:
        numpy.ndarray: Spherical coordinates :math:`(r, \theta_1, \theta_2)`
        in radians.
    """
    pts_cart = np.array(pts_cart)

    # Validate inputs
    is_one_point = False
    if pts_cart.shape == (3,):
        is_one_point = True
        pts_cart = pts_cart.reshape(1, 3)
    elif pts_cart.ndim != 2 or pts_cart.shape[1] != 3:
        raise ValueError("Shape of input must be either (3,) or (n, 3)")

    # Compute r
    r = np.sqrt(np.sum(np.square(pts_cart), axis=1))

    # Compute latitude
    z = pts_cart[:, 2]
    lat = np.arcsin(z / r)

    # Compute longitude
    x = pts_cart[:, 0]
    y = pts_cart[:, 1]
    lng = np.arctan2(y, x) # choosing the quadrant correctly

    # Assemble
    pts_r_lat_lng = np.stack((r, lat, lng), axis=-1)

    # Select output convention
    if convention == 'lat-lng':
        pts_sph = pts_r_lat_lng
    elif convention == 'theta-phi':
        pts_sph = _convert_sph_conventions(
            pts_r_lat_lng, 'lat-lng_to_theta-phi')
    else:
        raise NotImplementedError(convention)

    if is_one_point:
        pts_sph = pts_sph.reshape(3)

    return pts_sph


def _convert_sph_conventions(pts_r_angle1_angle2, what2what):
    """Internal function converting between different conventions for
    spherical coordinates. See :func:`cart2sph` for conventions.
    """
    if what2what == 'lat-lng_to_theta-phi':
        pts_r_theta_phi = np.zeros(pts_r_angle1_angle2.shape)
        # Radius is the same
        pts_r_theta_phi[:, 0] = pts_r_angle1_angle2[:, 0]
        # Angle 1
        pts_r_theta_phi[:, 1] = np.pi / 2 - pts_r_angle1_angle2[:, 1]
        # Angle 2
        ind = pts_r_angle1_angle2[:, 2] < 0
        pts_r_theta_phi[ind, 2] = 2 * np.pi + pts_r_angle1_angle2[ind, 2]
        pts_r_theta_phi[np.logical_not(ind), 2] = \
            pts_r_angle1_angle2[np.logical_not(ind), 2]
        return pts_r_theta_phi

    if what2what == 'theta-phi_to_lat-lng':
        pts_r_lat_lng = np.zeros(pts_r_angle1_angle2.shape)
        # Radius is the same
        pts_r_lat_lng[:, 0] = pts_r_angle1_angle2[:, 0]
        # Angle 1
        pts_r_lat_lng[:, 1] = np.pi / 2 - pts_r_angle1_angle2[:, 1]
        # Angle 2
        ind = pts_r_angle1_angle2[:, 2] > np.pi
        pts_r_lat_lng[ind, 2] = pts_r_angle1_angle2[ind, 2] - 2 * np.pi
        pts_r_lat_lng[np.logical_not(ind), 2] = \
            pts_r_angle1_angle2[np.logical_not(ind), 2]
        return pts_r_lat_lng

    raise NotImplementedError(what2what)


def sph2cart(pts_sph, convention='lat-lng'):
    """Inverse of :func:`cart2sph`.

    See :func:`cart2sph`.
    """
    pts_sph = np.array(pts_sph)

    # Validate inputs
    is_one_point = False
    if pts_sph.shape == (3,):
        is_one_point = True
        pts_sph = pts_sph.reshape(1, 3)
    elif pts_sph.ndim != 2 or pts_sph.shape[1] != 3:
        raise ValueError("Shape of input must be either (3,) or (n, 3)")

    # Degrees?
    _warn_degree(pts_sph[:, 1:])

    # Convert to latitude-longitude convention, if necessary
    if convention == 'lat-lng':
        pts_r_lat_lng = pts_sph
    elif convention == 'theta-phi':
        pts_r_lat_lng = _convert_sph_conventions(
            pts_sph, 'theta-phi_to_lat-lng')
    else:
        raise NotImplementedError(convention)

    # Compute x, y and z
    r = pts_r_lat_lng[:, 0]
    lat = pts_r_lat_lng[:, 1]
    lng = pts_r_lat_lng[:, 2]
    z = r * np.sin(lat)
    x = r * np.cos(lat) * np.cos(lng)
    y = r * np.cos(lat) * np.sin(lng)

    # Assemble and return
    pts_cart = np.stack((x, y, z), axis=-1)

    if is_one_point:
        pts_cart = pts_cart.reshape(3)

    return pts_cart



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def add_property2dict(target_dict, object, props):
    for prop in props:
        target_dict[prop] = getattr(object, prop)


def normalize(v, axis=0):
    # axis = 0, normalize each col
    # axis = 1, normalize each row
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-9)


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


def gen_render_path(c2ws, N_views=30):
    N = len(c2ws)
    rotvec, positions = [], []
    rotvec_inteplat, positions_inteplat = [], []
    weight = np.linspace(1.0, .0, N_views//3, endpoint=False).reshape(-1, 1)
    for i in range(N):
        r = R.from_matrix(c2ws[i, :3, :3])
        euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
        if i:
            mask = np.abs(euler_ange - rotvec[0])>180
            euler_ange[mask] += 360.0
        rotvec.append(euler_ange)
        positions.append(c2ws[i, :3, 3:].reshape(1, 3))

        if i:
            rotvec_inteplat.append(weight * rotvec[i - 1] + (1.0 - weight) * rotvec[i])
            positions_inteplat.append(weight * positions[i - 1] + (1.0 - weight) * positions[i])

    rotvec_inteplat.append(weight * rotvec[-1] + (1.0 - weight) * rotvec[0])
    positions_inteplat.append(weight * positions[-1] + (1.0 - weight) * positions[0])

    c2ws_render = []
    angles_inteplat, positions_inteplat = np.concatenate(rotvec_inteplat), np.concatenate(positions_inteplat)
    for rotvec, position in zip(angles_inteplat, positions_inteplat):
        c2w = np.eye(4)
        c2w[:3, :3] = R.from_euler('xyz', rotvec, degrees=True).as_matrix()
        c2w[:3, 3:] = position.reshape(3, 1)
        c2ws_render.append(c2w.copy())
    c2ws_render = np.stack(c2ws_render)
    return c2ws_render


def unique_lst(list1):
    x = np.array(list1)
    return np.unique(x)


def read_map(path):
    arr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if arr is None:
        raise RuntimeError(f"Failed to read\n\t{path}")
    # RGB
    if arr.ndim == 3 or arr.shape[2] == 3:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return rgb

    raise NotImplementedError(arr.shape)

def resize(arr, new_h=None, new_w=None, method='cv2'):
    """Resizes an image, with the option of maintaining the aspect ratio.

    Args:
        arr (numpy.ndarray): Image to binarize. If multiple-channel, each
            channel is resized independently.
        new_h (int, optional): Target height. If ``None``, will be calculated
            according to the target width, assuming the same aspect ratio.
        new_w (int, optional): Target width. If ``None``, will be calculated
            according to the target height, assuming the same aspect ratio.
        method (str, optional): Accepted values: ``'cv2'`` and ``'tf'``.

    Returns:
        numpy.ndarray: Resized image.
    """
    h, w = arr.shape[:2]
    if new_h is not None and new_w is not None:
        if int(h / w * new_w) != new_h:
            print((
                "Aspect ratio changed in resizing: original size is %s; "
                "new size is %s"), (h, w), (new_h, new_w))
    elif new_h is None and new_w is not None:
        new_h = int(h / w * new_w)
    elif new_h is not None and new_w is None:
        new_w = int(w / h * new_h)
    else:
        raise ValueError("At least one of new height or width must be given")

    if method in ('cv', 'cv2', 'opencv'):
        interp = cv2.INTER_LINEAR if new_h > h else cv2.INTER_AREA
        resized = cv2.resize(arr, (new_w, new_h), interpolation=interp)
    else:
        raise NotImplementedError(method)

    return resized

def write_uint(arr_uint, outpath):
    r"""Writes an ``uint`` array as an image to disk.

    Args:
        arr_uint (numpy.ndarray): A ``uint`` array.
        outpath (str): Output path.

    Writes
        - The resultant image.
    """
    if arr_uint.ndim == 3 and arr_uint.shape[2] == 1:
        arr_uint = np.dstack([arr_uint] * 3)

    img = Image.fromarray(arr_uint)
    img.save(outpath)

def write_arr(arr_0to1, outpath, img_dtype='uint8', clip=False):
    r"""Writes a ``float`` array as an image to disk.

    Args:
        arr_0to1 (numpy.ndarray): Array with values roughly :math:`\in [0,1]`.
        outpath (str): Output path.
        img_dtype (str, optional): Image data type. Defaults to ``'uint8'``.
        clip (bool, optional): Whether to clip values to :math:`[0,1]`.
            Defaults to ``False``.

    Writes
        - The resultant image.

    Returns:
        numpy.ndarray: The resultant image array.
    """
    arr_min, arr_max = arr_0to1.min(), arr_0to1.max()
    if clip:
        if arr_max > 1:
            print("Maximum before clipping: %f", arr_max)
        if arr_min < 0:
            print("Minimum before clipping: %f", arr_min)
        arr_0to1 = np.clip(arr_0to1, 0, 1)
    else:
        assert arr_min >= 0 and arr_max <= 1, \
            "Input should be in [0, 1], or allow it to be clipped"
    # Float array to image
    img_arr = (arr_0to1 * np.iinfo(img_dtype).max).astype(img_dtype)

    write_uint(img_arr, outpath)

    return img_arr


def load_light(envmap_path, envmap_inten=1., envmap_h=None, vis_path=None):
    if envmap_path == 'white':
        h = 16 if envmap_h is None else envmap_h
        envmap = np.ones((h, 2 * h, 3), dtype=float)

    elif envmap_path == 'point':
        h = 16 if envmap_h is None else envmap_h
        envmap = np.zeros((h, 2 * h, 3), dtype=float)
        i = -envmap.shape[0] // 4
        j = -int(envmap.shape[1] * 7 / 8)
        d = 2
        envmap[(i - d):(i + d), (j - d):(j + d), :] = 1

    else:
        envmap = read_map(envmap_path)

    # Optionally resize
    if envmap_h is not None:
        envmap = resize(envmap, new_h=envmap_h)

    # Scale by intensity
    envmap = envmap_inten * envmap

    # visualize the environment map in effect
    if vis_path is not None:
        write_arr(envmap, vis_path, clip=True)

    return envmap

def gen_light_xyz(envmap_h, envmap_w, envmap_radius=1e2):
    """Additionally returns the associated solid angles, for integration.
    """
    # OpenEXR "latlong" format
    # lat = pi/2
    # lng = pi
    #     +--------------------+
    #     |                    |
    #     |                    |
    #     +--------------------+
    #                      lat = -pi/2
    #                      lng = -pi
    lat_step_size = np.pi / (envmap_h + 2)
    lng_step_size = 2 * np.pi / (envmap_w)
    # Try to exclude the problematic polar points
    lats = np.linspace(
        np.pi / 2 - lat_step_size, -np.pi / 2 + lat_step_size, envmap_h)
    lngs = np.linspace(
        np.pi, -np.pi + lng_step_size, envmap_w)
    lngs, lats = np.meshgrid(lngs, lats)

    # To Cartesian
    rlatlngs = np.dstack((envmap_radius * np.ones_like(lats), lats, lngs))
    rlatlngs = rlatlngs.reshape(-1, 3)
    xyz = sph2cart(rlatlngs)
    xyz = xyz.reshape(envmap_h, envmap_w, 3)

    # Calculate the area of each pixel on the unit sphere (useful for
    # integration over the sphere)
    sin_colat = np.sin(np.pi / 2 - lats)
    areas = 4 * np.pi * sin_colat / np.sum(sin_colat)

    assert 0 not in areas, \
        "There shouldn't be light pixel that doesn't contribute"

    return xyz, areas

def compute_visibility(cam_depth, light_depth, uv, cam_K, light_K, 
        camrotc2w, cam_pos, lightrotw2c, light_pos, depth_thres=0.01, 
        soft_vis=True, dot_bias=False, normals=None):

    f_x, f_y = cam_K[0,0], cam_K[1,1]
    c_x, c_y = cam_K[0,2], cam_K[1,2]
    f_x_l, f_y_l = light_K[0,0], light_K[1,1]
    c_x_l, c_y_l = light_K[0,2], light_K[1,2]
    
    # pix_mask = (cam_depth>0).reshape(-1)# (light_depth_reproj[...,2]>8).reshape(-1)
    u, v = uv[...,0], uv[...,1]
    cam_depth_c = torch.stack([
        (u-c_x)/f_x * cam_depth, (v-c_y)/f_y * cam_depth, cam_depth
    ], -1)
    
    cam_depth_w = (cam_depth_c[...,None,:] * camrotc2w[:,None,...]).sum(-1) + cam_pos
    light_dir = cam_depth_w - light_pos[:,None]
    light_depth_reproj = (light_dir[...,None,:] * lightrotw2c[:,None,...]).sum(-1)
    
    depth_reproj = light_depth_reproj[...,2]
    uv_reproj = light_depth_reproj[...,:2] / depth_reproj[...,None]
    # rescale to [-1,1] for F.grid_sample
    uv_reproj[...,0] = (uv_reproj[...,0] * f_x_l) / c_x_l
    uv_reproj[...,1] = (uv_reproj[...,1] * f_y_l) / c_y_l
    sample_depth = F.grid_sample(
        light_depth[:,None,...,0], uv_reproj[:,None,...], 
        padding_mode="border", align_corners=True)[:,0,0,:] # align_corners=True is better.
    # shadow_bias
    soft_r = 1.
    if dot_bias:
        light_dir = F.normalize(light_dir, dim=-1)
        normals = F.normalize(normals, dim=-1)
        depth_thres = (depth_thres * (1 - (-light_dir * normals).sum(-1).clamp_min(0))).clamp_min(0.5 * depth_thres)
        # depth_thres = (depth_thres * (-light_dir * normals).sum(-1).acos().tan()).clamp(0.1*depth_thres, 2*depth_thres)
        # soft_r = 0.5
    if not soft_vis: # or dot_bias:
        visibility = ~((depth_reproj - sample_depth) > depth_thres)
    else:
        # visibility = 1 - (depth_reproj - sample_depth).clamp(0, depth_thres) / depth_thres
        if not dot_bias:
            visibility = 1 - (depth_reproj - sample_depth - depth_thres).clamp(0, depth_thres*soft_r)\
                    / (depth_thres*soft_r)
        else:
            depth_diff = (depth_reproj - sample_depth - depth_thres).clamp_min(0)
            visibility = 1- torch.where(depth_diff < depth_thres*soft_r, depth_diff, depth_thres*soft_r)\
                 / (depth_thres*soft_r)
    return visibility, light_dir, cam_depth_w


def main(func_name):
    """Unit tests that can also serve as example usage."""
    if func_name in ('sph2cart', 'cart2sph'):
        # cart2sph() and sph2cart()
        pts_car = np.array([
            [-1, 2, 3],
            [4, -5, 6],
            [3, 5, -8],
            [-2, -5, 2],
            [4, -2, -23]])
        print(pts_car)
        pts_sph = cart2sph(pts_car)
        print(pts_sph)
        pts_car_recover = sph2cart(pts_sph)
        print(pts_car_recover)
    else:
        raise NotImplementedError("Unit tests for %s" % func_name)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('func', type=str, help="function to test")
    args = parser.parse_args()

    main(args.func)
