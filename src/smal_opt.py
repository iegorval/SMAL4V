"""
Original author: Silvia Zuffi (https://github.com/silviazuffi/smalst)
Refactoring and Python 3 porting: Valeria Iegorova (https://github.com/iegorval/smalmv)
MIT licence
"""

import os
import pdb
import json
import imageio
import torch
import torchvision
import torch.nn as nn
import numpy as np
import pickle as pkl
from torch.autograd import Variable
from src import train_utils
from src import loss_utils
from src import geom_utils
from src import smal_mesh_net
from src.nmr import NeuralRenderer
from utils import image as image_utils
from utils import smal_vis
from utils import visutil
from utils.mesh import sample_texture
from utils.obj2nmr import obj2nmr_uvmap
from data import zebra as zebra_data
from smal_model.smal_basics import load_smal_model
from smal_model.smal_torch import SMAL
from collections import OrderedDict


# TODO: pass as flags
CONFIG_DIR = "configs/"
CONFIG_MODEL = "optimization_params.json"
CONFIG_DATA = "zebra_data.json"


class ShapeTrainer(train_utils.Trainer):
    def define_model(self):
        opts = self.opts
        self.symmetric = opts["symmetric"]
        img_size = (opts["img_size"], opts["img_size"])

        # TODO: replace hard-coded paths with parameters
        texture_mask_path = (
            opts["dataset"]
            + "_data/texture_maps/my_smpl_00781_4_all_template_w_tex_uv_001_mask_small.png"
        )
        self.texture_map_mask = torch.Tensor(
            imageio.imread(texture_mask_path) / 255.0
        ).cuda(device=opts["gpu_id"])

        tex_masks = None

        data_path = "smpl_models/my_smpl_data_00781_4_all.pkl"
        data = pkl.load(open(data_path, "rb"), encoding="latin1")

        pca_var = data["eigenvalues"][: opts["num_betas"]]
        self.betas_prec = (
            torch.Tensor(pca_var)
            .cuda(device=opts["gpu_id"])
            .expand(opts["batch_size"], opts["num_betas"])
        )

        self.model = smal_mesh_net.MeshNet(
            img_size,
            opts,
            nz_feat=opts["nz_feat"],
            num_kps=opts["num_kps"],
            tex_masks=tex_masks,
        )

        if opts["num_pretrain_epochs"] > 0:
            self.load_network(self.model, "pred", opts["num_pretrain_epochs"])

        self.model = self.model.cuda(device=opts["gpu_id"])

        if not opts["infer_vert2kp"]:
            vert2kp_fname = opts["dataset"] + "_data/verts2kp.pkl"
            self.vert2kp = torch.Tensor(
                pkl.load(open(vert2kp_fname, "rb"), encoding="latin1")
            ).cuda(device=opts["gpu_id"])

        # Data structures to use for triangle priors.
        edges2verts = self.model.edges2verts
        # B x E x 4
        edges2verts = np.tile(
            np.expand_dims(edges2verts, 0), (opts["batch_size"], 1, 1)
        )
        self.edges2verts = Variable(
            torch.LongTensor(edges2verts).cuda(device=opts["gpu_id"]),
            requires_grad=False,
        )
        # For renderering.
        faces = self.model.faces.view(1, -1, 3)
        self.faces = faces.repeat(opts["batch_size"], 1, 1)
        self.renderer = NeuralRenderer(
            opts["img_size"],
            opts["projection_type"],
            opts["norm_f"],
            opts["norm_z"],
            opts["norm_f0"],
        )

        if opts["texture"]:
            self.tex_renderer = NeuralRenderer(
                opts["img_size"],
                opts["projection_type"],
                opts["norm_f"],
                opts["norm_z"],
                opts["norm_f0"],
            )
            # Only use ambient light for tex renderer
            if opts["use_directional_light"]:
                self.tex_renderer.directional_light_only()
            else:
                self.tex_renderer.ambient_light_only()

        # For visualization
        self.vis_rend = smal_vis.VisRenderer(
            opts["img_size"],
            faces.data.cpu().numpy(),
            opts["projection_type"],
            opts["norm_f"],
            opts["norm_z"],
            opts["norm_f0"],
        )
        self.background_imgs = None

    def init_dataset(self):
        opts = self.opts
        if opts["dataset"] == "zebra":
            self.data_module = zebra_data
        else:
            print("Unknown dataset %d!" % opts["dataset"])

        self.dataloader = self.data_module.data_loader(opts)
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def define_criterion(self):
        opts = self.opts
        if opts["use_keypoints"]:
            self.projection_loss = loss_utils.kp_l2_loss
        if opts["use_mask"]:
            self.mask_loss_fn = loss_utils.mask_loss
        if opts["infer_vert2kp"]:
            self.entropy_loss = loss_utils.entropy_loss
        if self.opts["use_camera_loss"]:
            self.camera_loss = loss_utils.camera_loss

        if opts["use_smal_betas"]:
            self.betas_loss_fn = loss_utils.betas_loss
        self.delta_v_loss_fn = loss_utils.delta_v_loss

        if opts["texture"]:
            if opts["use_perceptual_loss"]:
                self.texture_loss = loss_utils.PerceptualTextureLoss()
            else:
                self.texture_loss = loss_utils.texture_loss
            self.texture_dt_loss_fn = loss_utils.texture_dt_loss
            if opts["texture_map"]:
                self.texture_map_loss = loss_utils.texture_map_loss
            if opts["uv_flow"]:
                self.uv_flow_loss = loss_utils.uv_flow_loss

        self.model_trans_loss_fn = loss_utils.model_trans_loss
        self.model_pose_loss_fn = loss_utils.model_pose_loss

    def set_optimization_input(self):
        opts = self.opts
        cams = np.zeros((self.scale_pred.shape[0], 3))
        cams[:, 0] = self.scale_pred.data.cpu()
        cams[:, 1:] = 128
        self.cams = Variable(
            torch.FloatTensor(cams).cuda(device=opts["gpu_id"]), requires_grad=False
        )
        self.model_trans = Variable(
            self.trans_pred.cuda(device=opts["gpu_id"]), requires_grad=False
        )

    def set_optimization_variables(self):
        """
        Sets as optimization variables those obtained as prediction from the network
        """
        opts = self.opts
        cams = np.zeros((self.scale_pred.shape[0], 3))
        cams[:, 0] = self.scale_pred.data
        cams[:, 1:] = 128

        # Prediction is gt
        self.cams = Variable(
            torch.FloatTensor(cams).cuda(device=opts["gpu_id"]), requires_grad=False
        )
        self.model_pose = Variable(
            self.pose_pred.cuda(device=opts["gpu_id"]), requires_grad=False
        )
        self.model_trans = Variable(
            self.trans_pred.cuda(device=opts["gpu_id"]), requires_grad=False
        )
        self.delta_v = Variable(
            self.delta_v.cuda(device=opts["gpu_id"]), requires_grad=False
        )

    def set_mask(self, batch):
        if "mask" in batch.keys():
            mask_tensor = batch["mask"].type(torch.FloatTensor)
            self.masks = Variable(
                mask_tensor.cuda(device=self.opts["gpu_id"]), requires_grad=False
            )
        else:
            self.masks = None

    def set_keypoints(self, batch):
        if self.opts["use_keypoints"] and "kp" in batch.keys():
            kp_tensor = batch["kp"].type(torch.FloatTensor)
            self.kps = Variable(
                kp_tensor.cuda(device=self.opts["gpu_id"]), requires_grad=False
            )
        else:
            self.kps = None

    def set_camera(self, batch):
        if "camera_params" in batch.keys():
            cam_tensor = batch["camera_params"].type(torch.FloatTensor)
            if self.opts["use_norm_f_and_z"]:
                cam_tensor[:, 0] = (
                    cam_tensor[:, 0] - self.opts["norm_f0"]
                ) / self.opts["norm_f"]
            self.cams = Variable(
                cam_tensor.cuda(device=self.opts["gpu_id"]), requires_grad=False
            )
        else:
            self.cams = None
            cam_c_tensor = batch["camera_params_c"].type(torch.FloatTensor)
            self.cams_center = Variable(
                cam_c_tensor.cuda(device=self.opts["gpu_id"]), requires_grad=False
            )

    def set_model_translation(self, batch):
        if "model_trans" in batch.keys():
            model_trans_tensor = batch["model_trans"].type(torch.FloatTensor)
            if self.opts["use_norm_f_and_z"]:
                model_trans_tensor[:, 2] = (
                    model_trans_tensor[:, 2] - self.opts["norm_z"] + 1.0
                )
            self.model_trans = Variable(
                model_trans_tensor.cuda(device=self.opts["gpu_id"]), requires_grad=False
            )

    def set_model_pose(self, batch):
        if "model_pose" in batch.keys():
            model_pose_tensor = batch["model_pose"].type(torch.FloatTensor)
            self.model_pose = Variable(
                model_pose_tensor.cuda(device=self.opts["gpu_id"]), requires_grad=False
            )
        else:
            self.model_trans = None
            self.model_pose = None

    def set_model_betas(self, batch):
        if "model_betas" in batch.keys():
            model_betas_tensor = batch["model_betas"][:, : self.opts["num_betas"]].type(
                torch.FloatTensor
            )
            self.model_betas = Variable(
                model_betas_tensor.cuda(device=self.opts["gpu_id"]), requires_grad=False
            )
        else:
            self.model_betas = None

    def set_model_delta_v(self, batch):
        if "model_delta_v" in batch.keys():
            model_delta_v_tensor = batch["model_delta_v"].type(torch.FloatTensor)
            self.model_delta_v = Variable(
                model_delta_v_tensor.cuda(device=self.opts["gpu_id"]),
                requires_grad=False,
            )
        else:
            self.model_delta_v = None

    def set_texture_map(self, batch):
        if self.opts["texture_map"]:
            assert "texture_map" in batch.keys()
            texture_map_tensor = batch["texture_map"].type(torch.FloatTensor)
            self.texture_map = Variable(
                texture_map_tensor.cuda(device=self.opts["gpu_id"]), requires_grad=False
            )
        else:
            self.texture_map = None

    def set_uv_flow(self, batch):
        if "uv_flow" in batch.keys():
            uv_flow_tensor = (
                batch["uv_flow"].type(torch.FloatTensor).permute(0, 3, 1, 2)
            )
            self.uv_flow_gt = Variable(
                uv_flow_tensor.cuda(device=self.opts["gpu_id"]), requires_grad=False
            )
        else:
            self.uv_flow_gt = None

    def compute_barrier_distance_transform(self, batch):
        if self.masks is not None:
            mask_dts = np.stack(
                [image_utils.compute_dt_barrier(m) for m in batch["mask"]]
            )
            dt_tensor = torch.FloatTensor(mask_dts).cuda(device=self.opts["gpu_id"])
            # B x 1 x N x N
            self.dts_barrier = Variable(dt_tensor, requires_grad=False).unsqueeze(1)

    def set_input(self, batch):
        # Image with annotations.
        input_img_tensor = batch["img"].type(torch.FloatTensor)

        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])

        img_tensor = batch["img"].type(torch.FloatTensor)
        self.input_imgs = Variable(
            input_img_tensor.cuda(device=self.opts["gpu_id"]), requires_grad=False
        )
        self.imgs = Variable(
            img_tensor.cuda(device=self.opts["gpu_id"]), requires_grad=False
        )
        self.img_paths = batch["img_path"]
        self.set_mask(batch)
        self.set_keypoints(batch)
        self.set_camera(batch)
        self.set_model_translation(batch)
        self.set_model_pose(batch)
        self.set_model_betas(batch)
        self.set_model_delta_v(batch)
        self.set_texture_map(batch)
        self.set_uv_flow(batch)
        self.compute_barrier_distance_transform(batch)

    def forward(
        self, opts_scale=None, opts_pose=None, opts_trans=None, opts_delta_v=None
    ):
        opts = self.opts
        if opts["use_double_input"]:
            masks = self.input_imgs * self.masks
        else:
            masks = None
        if opts["texture"]:
            pred_codes, self.textures = self.model.forward(self.input_imgs, masks)
        else:
            pred_codes = self.model.forward(self.input_imgs, masks)

        (
            self.delta_v,
            self.scale_pred,
            self.trans_pred,
            self.pose_pred,
            self.betas_pred,
            self.kp_2D_pred,
        ) = pred_codes

        if opts["fix_trans"]:
            self.trans_pred[:, 2] = self.model_trans[:, 2]

        if opts["use_gttrans"]:
            print("Using gt trans")
            self.trans_pred = self.model_trans
        if opts["use_gtpose"]:
            print("Using gt pose")
            self.pose_pred = self.model_pose
        if opts["use_gtcam"]:
            print("Using gt cam")
            self.scale_pred = self.cams[:, 0, None]
        if opts["use_gtbetas"]:
            print("Using gt betas")
            self.betas_pred = self.model_betas
        if opts["use_gtdeltav"]:
            print("Using gt delta_v")
            self.delta_v = self.model_delta_v

        if self.cams is not None:
            # The camera center does not change; here we predicting flength
            # print("Predicting flength:", self.scale_pred, self.cams[:, 1:])
            self.cam_pred = torch.cat([self.scale_pred, self.cams[:, 1:]], 1)
        else:
            # print("Camera center does change:", self.scale_pred, self.cams_center)
            self.cam_pred = torch.cat([self.scale_pred, self.cams_center], 1)

        if opts["only_mean_sym"]:
            del_v = self.delta_v
        else:
            del_v = self.model.symmetrize(self.delta_v)

        if opts["no_delta_v"]:
            del_v[:] = 0

        if opts["use_smal_pose"]:
            self.pred_v = self.model.get_smal_verts(
                self.pose_pred, self.betas_pred, self.trans_pred, del_v
            )
        else:
            # TODO: ?????
            self.mean_shape = self.model.get_mean_shape()
            self.pred_v = self.mean_shape + del_v + self.trans_pred

        # Compute keypoints.
        if opts["infer_vert2kp"]:
            self.vert2kp = torch.nn.functional.softmax(self.model.vert2kp, dim=1)
        self.kp_verts = torch.matmul(self.vert2kp, self.pred_v)

        # Set projection camera
        proj_cam = self.cam_pred
        # print(proj_cam)

        # Project keypoints
        if opts["use_keypoints"]:
            self.kp_pred = self.renderer.project_points(self.kp_verts, proj_cam)

        # Render mask.
        self.mask_pred = self.renderer.forward(self.pred_v, self.faces, proj_cam)

        if opts["texture"]:
            self.texture_flow = self.textures
            self.textures = geom_utils.sample_textures(self.texture_flow, self.imgs)
            tex_size = self.textures.size(2)
            self.textures = self.textures.unsqueeze(4).repeat(1, 1, 1, 1, tex_size, 1)

            if opts["use_gttexture"]:
                idx = 0
                uv_map = obj2nmr_uvmap(self.model.ft, self.model.vt, tex_size=tex_size)
                uv_img = self.texture_map[idx, :, :, :]
                uv_img = uv_img.permute(1, 2, 0)
                texture_t = sample_texture(uv_map, uv_img)
                self.textures[0, :, :, :, :, :] = texture_t[0, :, :, :, :, :]

            if opts["grad_v_in_tex_loss"]:
                self.texture_pred = self.tex_renderer.forward(
                    self.pred_v, self.faces, proj_cam.detach(), textures=self.textures
                )
            else:
                self.texture_pred = self.tex_renderer.forward(
                    self.pred_v.detach(),
                    self.faces,
                    proj_cam.detach(),
                    textures=self.textures,
                )
        else:
            self.textures = None
            if (
                opts["save_training_imgs"]
                and opts["use_mask"]
                and self.masks is not None
            ):
                T = 255 * self.mask_pred.cpu().detach().numpy()[0, :, :]
                imageio.imwrite(opts["name"] + "_mask_pred.png", T.astype(np.uint8))
                T = 255 * self.masks.cpu().detach().numpy()[0, :, :, :]
                T = np.transpose(T, (1, 2, 0))[:, :, 0]
                imageio.imwrite(opts["name"] + "_mask_gt.png", T.astype(np.uint8))

        # Compute losses for this instance.
        if self.opts["use_keypoints"] and self.kps is not None:
            self.kp_loss = self.projection_loss(self.kp_pred, self.kps)
        if self.opts["use_mask"] and self.masks is not None:
            self.mask_loss = self.mask_loss_fn(self.mask_pred, self.masks[:, 0, :, :])
        if self.opts["use_camera_loss"] and self.cams is not None:
            self.cam_loss = self.camera_loss(
                self.cam_pred, self.cams, 0, self.opts["use_norm_f_and_z"]
            )
        if self.model_trans is not None:
            self.mod_trans_loss = self.model_trans_loss_fn(
                self.trans_pred, self.model_trans
            )
        if self.model_pose is not None:
            self.mod_pose_loss = self.model_pose_loss_fn(
                self.pose_pred, self.model_pose, self.opts
            )

        # print(self.kp_loss, self.mask_loss, self.cam_loss, self.mod_trans_loss, self.mod_pose_loss)

        if opts["texture"]:
            if opts["use_loss_on_whole_image"]:
                if self.background_imgs is None:
                    print("SETTING BACKGROUND MODEL")
                    self.background_imgs = np.zeros(self.imgs.shape)
                    fg_mask = self.mask_pred.detach().cpu().numpy()
                    I = self.imgs.detach().cpu().numpy()
                    bg_mask = np.abs(fg_mask - 1)
                    rgb = np.zeros((3))
                    n = np.sum(bg_mask)
                    for c in range(3):
                        I[:, c, :, :] = I[:, c, :, :] * bg_mask
                        rgb[c] = np.sum(I[0, c, :, :]) / n

                    if self.background_model_top is not None:
                        N = 128
                        for c in range(3):
                            self.background_imgs[
                                :, c, :N, :
                            ] = self.background_model_top[c]
                            self.background_imgs[
                                :, c, N:, :
                            ] = self.background_model_bottom[c]
                    else:
                        # This is what we use for optimization
                        if opts["use_per_image_rgb_bg"]:
                            self.background_imgs[:, 0, :, :] = rgb[0]
                            self.background_imgs[:, 1, :, :] = rgb[1]
                            self.background_imgs[:, 2, :, :] = rgb[2]
                        else:
                            self.background_imgs[:, 0, :, :] = 0.6964
                            self.background_imgs[:, 1, :, :] = 0.5806
                            self.background_imgs[:, 2, :, :] = 0.4780

                        # Verification experiment: replace with image
                        if opts["use_img_as_background"]:
                            self.background_imgs[:, 0, :, :] = self.imgs.data[
                                :, 0, :, :
                            ]
                            self.background_imgs[:, 1, :, :] = self.imgs.data[
                                :, 1, :, :
                            ]
                            self.background_imgs[:, 2, :, :] = self.imgs.data[
                                :, 2, :, :
                            ]

                    self.background_imgs = torch.Tensor(self.background_imgs).cuda(
                        device=opts["gpu_id"]
                    )
            if self.masks is not None:
                if opts["use_loss_on_whole_image"]:
                    self.tex_loss = self.texture_loss(
                        self.texture_pred,
                        self.imgs,
                        self.mask_pred,
                        None,
                        self.background_imgs,
                    )
                else:
                    self.tex_loss = self.texture_loss(
                        self.texture_pred,
                        self.imgs,
                        self.mask_pred,
                        self.masks[:, 0, :, :],
                    )
                if opts["use_tex_dt"]:
                    self.tex_dt_loss = self.texture_dt_loss_fn(
                        self.texture_flow, self.dts_barrier[:, :, :, :, 0]
                    )
                # print("tex_loss and tex_dt_loss:", self.tex_loss, self.tex_dt_loss)
            else:
                if opts["use_loss_on_whole_image"]:
                    self.tex_loss = self.texture_loss(
                        self.texture_pred,
                        self.imgs,
                        self.mask_pred,
                        None,
                        self.background_imgs,
                    )
                else:
                    self.tex_loss = self.texture_loss(
                        self.texture_pred, self.imgs, self.mask_pred, None
                    )
            if opts["texture_map"] and self.texture_map is not None:
                uv_flows = self.model.texture_predictor.uvimage_pred
                uv_flows = uv_flows.permute(0, 2, 3, 1)
                uv_images = torch.nn.functional.grid_sample(
                    self.imgs, uv_flows, align_corners=True
                )
                self.tex_map_loss = self.texture_map_loss(
                    uv_images, self.texture_map, self.texture_map_mask, self.opts
                )
                # print("tex_map_loss:", self.tex_map_loss)
            if opts["uv_flow"] and self.uv_flow_gt is not None:
                uv_flows = self.model.texture_predictor.uvimage_pred
                self.uv_f_loss = self.uv_flow_loss(uv_flows, self.uv_flow_gt)
                # print("uv_f_loss:", self.uv_f_loss)
        # Priors:
        if opts["infer_vert2kp"]:
            self.vert2kp_loss = self.entropy_loss(self.vert2kp)
        if opts["use_smal_betas"]:
            self.betas_loss = self.betas_loss_fn(
                self.betas_pred, self.model_betas, self.betas_prec
            )
        if self.model_delta_v is not None:
            self.delta_v_loss = self.delta_v_loss_fn(self.delta_v, self.model_delta_v)
            # print("delta_v_loss:", self.delta_v_loss)

        # Finally sum up the loss.
        # Instance loss:
        if opts["use_keypoints"] and self.kps is not None:
            # print("Using keypoints:", self.kps is None, opts['kp_loss_wt'], self.kp_loss)
            self.total_loss = opts["kp_loss_wt"] * self.kp_loss
            if opts["use_mask"] and self.masks is not None:
                # print("And using mask:", self.masks is None, opts["mask_loss_wt"], self.mask_loss)
                self.total_loss += opts["mask_loss_wt"] * self.mask_loss
        else:
            if opts["use_mask"] and self.masks is not None:
                self.total_loss = opts["mask_loss_wt"] * self.mask_loss
            else:
                self.total_loss = 0

        if (
            not opts["use_gtcam"]
            and self.opts["use_camera_loss"]
            and self.cams is not None
        ):
            self.total_loss += opts["cam_loss_wt"] * self.cam_loss

        if opts["texture"]:
            self.total_loss += opts["tex_loss_wt"] * self.tex_loss

        if opts["texture_map"] and self.texture_map is not None:
            self.total_loss += opts["tex_map_loss_wt"] * self.tex_map_loss
        if opts["uv_flow"] and self.uv_flow_gt is not None:
            self.total_loss += opts["uv_flow_loss_wt"] * self.uv_f_loss
        if self.model_trans is not None:
            if not opts["use_gttrans"]:
                self.total_loss += opts["mod_trans_loss_wt"] * self.mod_trans_loss
        if self.model_pose is not None:
            if not opts["use_gtpose"]:
                self.total_loss += opts["mod_pose_loss_wt"] * self.mod_pose_loss

        if self.model_delta_v is not None:
            self.total_loss += opts["delta_v_loss_wt"] * self.delta_v_loss

        # Priors:
        if opts["infer_vert2kp"]:
            self.total_loss += opts["vert2kp_loss_wt"] * self.vert2kp_loss
        if opts["use_smal_betas"]:
            self.total_loss += opts["betas_reg_wt"] * self.betas_loss

        if opts["texture"] and self.masks is not None and opts["use_tex_dt"]:
            self.total_loss += opts["tex_dt_loss_wt"] * self.tex_dt_loss

        # print('---------------------')

    def get_current_visuals(self):
        vis_dict = {}
        try:
            mask_concat = torch.cat([self.masks[:, 0, :, :], self.mask_pred], 2)
        except:
            pdb.set_trace()

        if self.opts["texture"]:
            # B x 2 x H x W
            uv_flows = self.model.texture_predictor.uvimage_pred
            # B x H x W x 2
            uv_flows = uv_flows.permute(0, 2, 3, 1)
            uv_images = torch.nn.functional.grid_sample(
                self.imgs, uv_flows, align_corners=True
            )

        num_show = min(2, self.opts["batch_size"])
        show_uv_imgs = []
        show_uv_flows = []

        for i in range(num_show):
            input_img = smal_vis.kp2im(self.kps[i].data, self.imgs[i].data)
            pred_kp_img = smal_vis.kp2im(self.kp_pred[i].data, self.imgs[i].data)
            masks = smal_vis.tensor2mask(mask_concat[i].data)
            if self.opts["texture"]:
                texture_here = self.textures[i]
            else:
                texture_here = None

            rend_predcam = self.vis_rend(
                self.pred_v[i], self.cam_pred[i], texture=texture_here
            )
            # Render from front & back:
            rend_frontal = self.vis_rend.diff_vp(
                self.pred_v[i],
                self.cam_pred[i],
                texture=texture_here,
                kp_verts=self.kp_verts[i],
            )
            rend_top = self.vis_rend.diff_vp(
                self.pred_v[i],
                self.cam_pred[i],
                axis=[0, 1, 0],
                texture=texture_here,
                kp_verts=self.kp_verts[i],
            )
            diff_rends = np.hstack((rend_frontal, rend_top))

            if self.opts["texture"]:
                uv_img = smal_vis.tensor2im(uv_images[i].data)
                show_uv_imgs.append(uv_img)
                uv_flow = smal_vis.visflow(uv_flows[i].data)
                show_uv_flows.append(uv_flow)

                tex_img = smal_vis.tensor2im(self.texture_pred[i].data)
                imgs = np.hstack((input_img, pred_kp_img, tex_img))
            else:
                imgs = np.hstack((input_img, pred_kp_img))

            rend_gtcam = self.vis_rend(
                self.pred_v[i], self.cams[i], texture=texture_here
            )
            rends = np.hstack((diff_rends, rend_predcam, rend_gtcam))
            vis_dict["%d" % i] = np.hstack((imgs, rends, masks))
            vis_dict["masked_img %d" % i] = smal_vis.tensor2im(
                (self.imgs[i] * self.masks[i]).data
            )

        if self.opts["texture"]:
            vis_dict["uv_images"] = np.hstack(show_uv_imgs)
            vis_dict["uv_flow_vis"] = np.hstack(show_uv_flows)

        return vis_dict

    def get_current_points(self):
        return {
            "mean_shape": visutil.tensor2verts(self.mean_shape.data),
            "verts": visutil.tensor2verts(self.pred_v.data),
        }

    def get_current_scalars(self):
        sc_dict = OrderedDict(
            [
                ("smoothed_total_loss", self.smoothed_total_loss),
                ("total_loss", self.total_loss.item()),
            ]
        )
        if self.opts["use_smal_betas"]:
            sc_dict["betas_reg"] = self.betas_loss.item()
        if self.opts["use_mask"] and self.masks is not None:
            sc_dict["mask_loss"] = self.mask_loss.item()
        if self.opts["use_keypoints"] and self.kps is not None:
            sc_dict["kp_loss"] = self.kp_loss.item()
        if self.opts["use_camera_loss"] and self.cams is not None:
            sc_dict["cam_loss"] = self.cam_loss.item()
        if self.opts["texture"]:
            sc_dict["tex_loss"] = self.tex_loss.item()
        if (
            self.opts["texture_map"]
            and self.opts["use_tex_dt"]
            and self.masks is not None
        ):
            sc_dict["tex_dt_loss"] = self.tex_dt_loss.item()
        if self.opts["uv_flow"] and self.uv_flow_gt is not None:
            sc_dict["uv_flow_loss"] = self.uv_f_loss.item()
        if self.opts["texture_map"] and self.texture_map is not None:
            sc_dict["tex_map_loss"] = self.tex_map_loss.item()
        if self.model_trans is not None:
            sc_dict["model_trans_loss"] = self.mod_trans_loss.item()
        if self.model_pose is not None:
            sc_dict["model_pose_loss"] = self.mod_pose_loss.item()
        if self.opts["infer_vert2kp"]:
            sc_dict["vert2kp_loss"] = self.vert2kp_loss.item()
        if self.model_delta_v is not None:
            sc_dict["model_delta_v_loss"] = self.delta_v_loss.item()

        return sc_dict


def main(opts):
    torch.manual_seed(0)
    np.random.seed(0)
    print("Training initialization...")
    trainer = ShapeTrainer(opts)
    trainer.init_training()
    print("Done initializing training.")
    trainer.train()


if __name__ == "__main__":
    fname_data = os.path.join(CONFIG_DIR, CONFIG_DATA)
    with open(fname_data) as f:
        opts_data = json.load(f)
    opts_data["zebra_dir"] = "data/results/input_proc/"
    opts_data[
        "image_file_string"
    ] = "proc_0997705e-b62a-4e67-a26f-1003dc9c9885_male_3e8faee7-14f6-43ab-b0ad-4125bcd83d50_backright.jpg"
    fname_model = os.path.join(CONFIG_DIR, CONFIG_MODEL)
    with open(fname_model) as f:
        opts_model = json.load(f)
    opts = opts_data | opts_model
    main(opts)
