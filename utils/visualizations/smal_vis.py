"""
Visualization helpers specific to birds.
"""
import numpy as np
import cv2
import pylab
import torch
from torch.autograd import Variable
from matplotlib import cm
from src.nmr import NeuralRenderer


class VisRenderer:
    """
    Utility to render meshes using pytorch NMR
    faces are F x 3 or 1 x F x 3 numpy
    """

    def __init__(self, img_size, faces, proj_type, norm_f, norm_z, norm_f0, t_size=3):
        self.renderer = NeuralRenderer(img_size, proj_type, norm_f, norm_z, norm_f0)
        # By default white background
        self.set_bgcolor([1.0, 1.0, 1.0])
        self.faces = Variable(torch.IntTensor(faces).cuda(), requires_grad=False)
        if self.faces.dim() == 2:
            self.faces = torch.unsqueeze(self.faces, 0)
        default_tex = np.ones((1, self.faces.shape[1], t_size, t_size, t_size, 3))

        blue = np.array([101, 142, 162.0]) / 255.0
        # Color for optimization results
        # blue = np.array([234, 156, 199.]) / 255. # purple for optimization

        default_tex = default_tex * blue
        # Could make each triangle different color
        self.default_tex = Variable(
            torch.FloatTensor(default_tex).cuda(), requires_grad=False
        )

        cam = np.array([2000.0, img_size / 2.0, img_size / 2.0])
        self.default_cam = Variable(torch.FloatTensor(cam).cuda(), requires_grad=False)
        self.default_cam = torch.unsqueeze(self.default_cam, 0)

    def __call__(self, verts, gpu_id, cams=None, texture=None, rend_mask=False):
        """
        verts is |V| x 3 cuda torch Variable
        cams is 7, cuda torch Variable
        Returns N x N x 3 numpy
        """
        if texture is None:
            texture = self.default_tex
        elif texture.dim() == 5:
            # Here input it F x T x T x T x 3 (instead of F x T x T x 3)
            # So add batch dim.
            texture = torch.unsqueeze(texture, 0)
        if cams is None:
            cams = self.default_cam
        elif cams.dim() == 1:
            cams = torch.unsqueeze(cams, 0)

        if verts.dim() == 2:
            verts = torch.unsqueeze(verts, 0)

        verts = as_variable(verts)
        cams = as_variable(cams)
        texture = as_variable(texture)

        if rend_mask:
            rend = self.renderer.forward(verts, self.faces, cams, gpu_id)
            rend = rend.repeat(3, 1, 1)
            rend = rend.unsqueeze(0)
        else:
            rend = self.renderer.forward(verts, self.faces, cams, gpu_id, texture)

        rend = rend.data.cpu().numpy()[0].transpose((1, 2, 0))
        rend = np.clip(rend, 0, 1) * 255.0

        return rend.astype(np.uint8)

    def rotated(self, vert, deg, axis=(0, 1, 0), cam=None, texture=None):
        """
        vert is N x 3, torch FloatTensor (or Variable)
        """
        new_rot = cv2.Rodrigues(np.deg2rad(deg) * np.array(axis))[0]
        new_rot = convert_as(torch.FloatTensor(new_rot), vert)

        center = vert.mean(0)
        new_vert = torch.t(torch.matmul(new_rot, torch.t(vert - center))) + center

        return self.__call__(new_vert, cams=cam, texture=texture)

    def set_bgcolor(self, color):
        self.renderer.set_bgcolor(color)

    def set_light_dir(self, direction, int_dir=0.8, int_amb=0.8):
        renderer = self.renderer.renderer
        renderer.light_direction = direction
        renderer.light_intensity_directional = int_dir
        renderer.light_intensity_ambient = int_amb


def as_variable(x):
    if type(x) is not torch.autograd.Variable:
        x = Variable(x, requires_grad=False)
    return x


def convert_as(src, trg):
    src = src.type_as(trg)
    if src.is_cuda:
        src = src.cuda(device=trg.get_device())
    if type(trg) is torch.autograd.Variable:
        src = Variable(src, requires_grad=False)
    return src


def convert2np(x):
    # Assumes x is gpu tensor..
    if type(x) is not np.ndarray:
        return x.cpu().numpy()
    return x


def tensor2mask(image_tensor, imtype=np.uint8):
    # Input is H x W
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.expand_dims(image_numpy, 2) * 255.0
    image_numpy = np.tile(image_numpy, (1, 1, 3))
    return image_numpy.astype(imtype)


def kp2im(kp, img, radius=None):
    """
    Input is numpy array or torch.cuda.Tensor
    img can be H x W, H x W x C, or C x H x W
    kp is |KP| x 2

    """
    kp_norm = convert2np(kp)
    img = convert2np(img)

    if img.ndim == 2:
        img = np.dstack((img,) * 3)
    # Make it H x W x C:
    elif img.shape[0] == 1 or img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:  # Gray2RGB for H x W x 1
            img = np.dstack((img,) * 3)

    # kp_norm is still in [-1, 1], converts it to image coord.
    kp = (kp_norm[:, :2] + 1) * 0.5 * img.shape[0]
    if kp_norm.shape[1] == 3:
        vis = kp_norm[:, 2] > 0
        kp[~vis] = 0
        kp = np.hstack((kp, vis.reshape(-1, 1)))
    else:
        vis = np.ones((kp.shape[0], 1))
        kp = np.hstack((kp, vis))

    kp_img = draw_kp(kp, img, radius=radius)

    return kp_img


def draw_kp(kp, img, radius=None):
    """
    kp is 15 x 2 or 3 numpy.
    img can be either RGB or Gray
    Draws bird points.
    """
    if radius is None:
        radius = max(4, (np.mean(img.shape[:2]) * 0.01).astype(int))

    num_kp = kp.shape[0]
    # Generate colors
    cm = pylab.get_cmap("gist_rainbow")
    colors = 255 * np.array([cm(1.0 * i / num_kp)[:3] for i in range(num_kp)])
    white = np.ones(3) * 255

    image = img.copy()

    if isinstance(image.reshape(-1)[0], np.float32):
        # Convert to 255 and np.uint8 for cv2..
        image = (image * 255).astype(np.uint8)

    kp = np.round(kp).astype(int)

    for kpi, color in zip(kp, colors):
        # This sometimes causes OverflowError,,
        if kpi[2] == 0:
            continue
        cv2.circle(image, (kpi[0], kpi[1]), radius + 1, white, -1)
        cv2.circle(image, (kpi[0], kpi[1]), radius, color, -1)
    return image


def tensor2im(image_tensor, imtype=np.uint8, scale_to_range_1=False):
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    if scale_to_range_1:
        image_numpy = image_numpy - np.min(image_numpy, axis=2, keepdims=True)
        image_numpy = image_numpy / np.max(image_numpy)
    else:
        # Clip to [0, 1]
        image_numpy = np.clip(image_numpy, 0, 1)

    return (image_numpy * 255).astype(imtype)


def visflow(flow_img):
    # H x W x 2
    flow_img = convert2np(flow_img)

    x_img = flow_img[:, :, 0]

    def color_within_01(vals):
        # vals is Nx1 in [-1, 1] (but could be larger)
        vals = np.clip(vals, -1, 1)
        # make [0, 1]
        vals = (vals + 1) / 2.0
        # Append dummy end vals for consistent coloring
        weights = np.hstack([vals, np.array([0, 1])])
        # Drop the dummy colors
        colors = cm.plasma(weights)[:-2, :3]
        return colors

    x_color = color_within_01(x_img.reshape(-1))
    x_color = x_color.reshape([x_img.shape[0], x_img.shape[1], 3])
    y_img = flow_img[:, :, 1]
    y_color = color_within_01(y_img.reshape(-1))
    y_color = y_color.reshape([y_img.shape[0], y_img.shape[1], 3])
    vis = np.vstack([x_color, y_color])
    return vis
