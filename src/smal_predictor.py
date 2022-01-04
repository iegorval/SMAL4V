"""
Takes an image, returns stuff.

Original author: Silvia Zuffi (https://github.com/silviazuffi/smalst)
Refactoring and Python 3 porting: Valeria Iegorova (https://github.com/iegorval/smalmv)
MIT licence
"""
import os
import cv2
import pdb
import torch.nn as nn
import numpy as np
import pickle as pkl
import torch
import torchvision
from torch.autograd import Variable
from src import smal_mesh_net as mesh_net
from src import geom_utils
from src.nmr import NeuralRenderer
from utils import smal_vis
from smal_model.smal_torch import SMAL


class MeshPredictor:
    def __init__(self, opts):
        self.opts = opts
        self.symmetric = opts["symmetric"]

        # Load the texture map layers
        tex_masks = [None] * opts["number_of_textures"]
        self.vert2kp = torch.Tensor(
            pkl.load(open(opts["verts2kp_path"], "rb"), encoding="latin1")
        ).cuda(device=opts["gpu_id"])

        print("Setting up pipeline...")
        if opts["pipeline_version"] == "smalst":
            self.model = mesh_net.MeshNetSMALST(opts, tex_masks=tex_masks)
        elif opts["pipeline_version"] == "smbld":
            self.model = nn.DataParallel(mesh_net.MeshNet(opts))
        else:
            raise "Unsupported Pipeline"

        self.load_pretrained_network()  # TODO: if
        self.model.eval()
        self.model = self.model.cuda(device=self.opts["gpu_id"])

        self.renderer = NeuralRenderer(
            opts["img_size"],
            opts["projection_type"],
            opts["norm_f"],
            opts["norm_z"],
            opts["norm_f0"],
        )
        self.renderer.directional_light_only()

        if opts["texture"]:
            self.tex_renderer = NeuralRenderer(
                opts["img_size"],
                opts["projection_type"],
                opts["norm_f"],
                opts["norm_z"],
                opts["norm_f0"],
            )
            # Only use ambient light for tex renderer
            self.tex_renderer.ambient_light_only()

        # self.mean_shape = self.model.get_mean_shape()
        # Benjamin
        self.smal = SMAL(opts)
        faces = self.smal.faces.view(1, -1, 3)

        # For visualization
        # faces = self.model.faces.view(1, -1, 3)
        self.faces = faces.repeat(opts["batch_size"], 1, 1)
        self.vis_rend = smal_vis.VisRenderer(
            opts["img_size"],
            faces.data.cpu().numpy(),
            opts["projection_type"],
            opts["norm_f"],
            opts["norm_z"],
            opts["norm_f0"],
        )
        self.vis_rend.set_bgcolor([1.0, 1.0, 1.0])

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def load_pretrained_network(self):
        # save_filename = "{}_net_{}.pth".format(network_label, epoch_label)
        # network_dir = os.path.join(self.opts["checkpoint_dir"], self.opts["name"])
        # save_path = os.path.join(network_dir, save_filename)
        pretrained_path = os.path.join(
            self.opts["pretrained_dir"], self.opts["pretrained_name"]
        )
        state_dict = torch.load(pretrained_path)
        if self.opts["pipeline_version"] == "smbld":
            new_state_dict = {}
            for key in state_dict.keys():
                if (
                    "betas_scale_predictor" in key
                    and not self.opts["use_scaling_betas"]
                ):
                    continue
                new_key = key.replace(".netG_DETAIL", "").replace("model.", "")
                new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict = state_dict
        self.model.load_state_dict(new_state_dict)

    def set_input(self, batch):
        opts = self.opts

        # original image where texture is sampled from.
        img_tensor = batch["img"].clone().type(torch.FloatTensor)

        # input_img is the input to resnet
        input_img_tensor = batch["img"].type(torch.FloatTensor)

        # for b in range(input_img_tensor.size(0)):
        #     input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])

        self.input_imgs = Variable(
            input_img_tensor.cuda(device=opts["gpu_id"]), requires_grad=False
        )
        self.imgs = Variable(
            img_tensor.cuda(device=opts["gpu_id"]), requires_grad=False
        )

    def predict(
        self, batch, cam_gt=None, trans_gt=None, pose_gt=None, betas_gt=None, rot=0
    ):
        """
        batch has B x C x H x W numpy
        """
        self.set_input(batch)
        self.forward(cam_gt, trans_gt, pose_gt, betas_gt, rot)
        return self.collect_outputs()

    def forward(self, cam_gt=None, trans_gt=None, pose_gt=None, betas_gt=None, rot=0):
        if self.opts["texture"]:
            pred_codes, self.textures = self.model.forward(self.input_imgs)
        else:
            pred_codes, _ = self.model.forward(self.input_imgs)

        (
            self.delta_v,
            self.scale_pred,
            self.trans_pred,
            self.pose_pred,
            self.betas_pred,
            self.betas_scale_pred,
        ) = pred_codes

        # # Rotate the view
        # if rot != 0:
        #     r0 = self.pose_pred[:, :3].detach().cpu().numpy()
        #     R0, _ = cv2.Rodrigues(r0)
        #     ry = np.array([0, rot, 0])
        #     Ry, _ = cv2.Rodrigues(ry)
        #     Rt = np.matrix(Ry) * np.matrix(R0)
        #     rt, _ = cv2.Rodrigues(Rt)
        #     self.pose_pred[:, :3] = torch.Tensor(rt).permute(1, 0)

        if cam_gt is not None:
            print("Setting gt cam")
            self.scale_pred[:] = cam_gt
        if trans_gt is not None:
            print("Setting gt trans")
            self.trans_pred[0, :] = torch.Tensor(trans_gt)
        if pose_gt is not None:
            print("Setting gt pose")
            self.pose_pred[0, :] = torch.Tensor(pose_gt)
        if betas_gt is not None:
            print("Setting gt betas")
            self.betas_pred[0, :] = torch.Tensor(betas_gt[:10])
            print("Removing delta_v")
            self.delta_v[:] = 0

        # TODO: uncomment
        # if self.opts["projection_type"] == "perspective":
        #     # The camera center does not change;
        #     cam_center = torch.Tensor(
        #         [self.input_imgs.shape[2] // 2, self.input_imgs.shape[3] // 2]
        #     ).cuda(device=self.opts["gpu_id"])
        #     if scale.shape[0] == 1:
        #         self.cam_pred = torch.cat([scale, cam_center[None, :]], 1)
        #     else:
        #         self.cam_pred = torch.cat(
        #             [
        #                 scale.permute(1, 0),
        #                 cam_center.repeat(scale.shape[0], 1).permute(1, 0),
        #             ]
        #         ).permute(1, 0)
        # else:
        #     pdb.set_trace()

        # Benjamin
        self.cam_pred = torch.cat(
            [
                self.scale_pred[:, [0]],
                torch.ones(self.opts["batch_size"], 2).cuda()
                * self.opts["img_size"]
                / 2,
            ],
            dim=1,
        )

        # del_v = self.delta_v
        # Deform mean shape:  TODO: it is now None instead
        # if self.opts["ignore_pred_delta_v"]:
        #     del_v[:] = 0

        # if self.opts["use_smal_pose"]:  # TODO: why this if?
        # self.smal_verts = self.model.get_smal_verts(
        #     self.pose, self.betas, self.trans, del_v
        # )
        # self.pred_v = self.smal_verts
        self.verts, _, _ = self.smal(
            self.betas_pred,
            self.pose_pred,
            trans=self.trans_pred,
            betas_logscale=self.betas_scale_pred,
            # del_v=self.delta_v,
        )  # TODO: verts to verts_pred?

        # Benjamin
        self.faces = self.smal.faces.unsqueeze(0).expand(self.verts.shape[0], 7774, 3)
        # TODO: rewrite as constant 7774
        # self.renderer.directional_light_only()
        synth_rgb, synth_silhouettes = self.renderer(
            self.verts, self.faces, self.cam_pred, self.opts["gpu_id"]
        )
        synth_rgb = torch.clamp(synth_rgb, 0.0, 1.0)
        synth_silhouettes = synth_silhouettes.unsqueeze(1)
        # TODO: does it extract correct vertices?
        self.kp_verts = torch.matmul(self.vert2kp, self.verts)

        # Project keypoints
        self.kp_pred = self.renderer.project_points(self.kp_verts, self.cam_pred)
        # self.mask_pred = self.renderer.forward(
        _, self.mask_pred = self.renderer.forward(
            self.verts,
            self.faces,
            self.cam_pred,
            self.opts["gpu_id"],
        )

        # Render texture.
        if self.opts["texture"]:
            if self.textures.size(-1) == 2:
                # Flow texture!
                self.texture_flow = self.textures
                self.textures = geom_utils.sample_textures(self.textures, self.imgs)
            if self.textures.dim() == 5:  # B x F x T x T x 3
                tex_size = self.textures.size(2)
                self.textures = self.textures.unsqueeze(4).repeat(
                    1, 1, 1, 1, tex_size, 1
                )
            # Render texture:
            self.texture_pred = self.tex_renderer.forward(
                self.verts,
                self.faces,
                self.cam_pred,
                self.opts["gpu_id"],
                textures=self.textures,
            )
            # B x 2 x H x W
            uv_flows = self.model.texture_predictor.uvimage_pred
            # B x H x W x 2
            self.uv_flows = uv_flows.permute(0, 2, 3, 1)

            self.uv_images = torch.nn.functional.grid_sample(
                self.imgs, self.uv_flows, align_corners=True
            )
        else:
            self.textures = None

    def collect_outputs(self):
        outputs = {
            "pose_pred": self.pose_pred.data,
            "kp_pred": self.kp_pred.data,
            "verts": self.verts.data,
            "kp_verts": self.kp_verts.data,
            "cam_pred": self.cam_pred.data,
            "mask_pred": self.mask_pred.data,
            "trans_pred": self.trans_pred.data,
            # "kp_2D_pred": self.kp_2D_pred,
            "f": self.faces,
            "v": self.verts,  # TODO: is it really the same?
        }
        if self.opts["use_shape_predictor"]:
            outputs["shape_f"] = self.model.code_predictor.shape_predictor.shape_f.data
        if not self.opts[
            "ignore_pred_delta_v"
        ]:  # TODO: unify the delta_v-related parameters
            outputs["delta_v_pred"] = self.delta_v.data
        if self.opts["use_smal_betas"]:
            outputs["betas_pred"] = self.betas_pred.data
        if self.opts["texture"]:
            outputs["texture"] = self.textures
            outputs["texture_pred"] = self.texture_pred.data
            outputs["uv_image"] = self.uv_images.data
            outputs["uv_flow"] = self.uv_flows.data

        return outputs
