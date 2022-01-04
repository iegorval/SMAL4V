import os
import pickle as pkl
from posixpath import join
import torch
import torch.nn as nn
import numpy as np
import smal_mesh_net as mesh_net
from src.nmr import NeuralRenderer
from smal_model.smal_torch import SMAL
from utils import smal_vis


class MeshPredictor(nn.Module):
    def __init__(self, opts, is_train=True):
        super(MeshPredictor, self).__init__()
        self.opts = opts

        # Load the texture map layers
        tex_masks = [None] * opts["number_of_textures"]
        if "verts2kp_path" in self.opts:  # TODO: change to config or not?
            self.vert2kp = torch.Tensor(
                pkl.load(open(opts["verts2kp_path"], "rb"), encoding="latin1")
            ).cuda(device=opts["gpu_id"])
            self.joints2kp = None
        else:
            self.vert2kp = None
            self.joint2kp = np.array(opts["model_joints"])

        print("Setting up pipeline...")
        if opts["pipeline_version"] == "smalst":
            self.model = mesh_net.MeshNetSMALST(opts, tex_masks=tex_masks)
            change_proj_points = False
        elif opts["pipeline_version"] == "smbld":
            self.model = nn.DataParallel(mesh_net.MeshNet(opts))
            change_proj_points = True
        else:
            raise "Unsupported Pipeline"

        if (is_train and opts["use_pretrained"]) or not is_train:
            self.load_pretrained_network()

        self.model = self.model.cuda(device=self.opts["gpu_id"])

        self.renderer = NeuralRenderer(
            opts["img_size"],
            opts["projection_type"],
            opts["norm_f"],
            opts["norm_z"],
            opts["norm_f0"],
            change_proj_points,
        )
        self.renderer.directional_light_only()

        self.smal = SMAL(opts)
        faces = self.smal.faces.view(1, -1, 3)

        self.faces = faces.repeat(1, 1, 1)  # TODO remove & change shape
        self.vis_renderer = smal_vis.VisRenderer(
            opts["img_size"],
            faces.data.cpu().numpy(),
            opts["projection_type"],
            opts["norm_f"],
            opts["norm_z"],
            opts["norm_f0"],
        )
        self.vis_renderer.set_bgcolor([1.0, 1.0, 1.0])
        self.vis_renderer.set_light_dir([0, 1, -1], 0.4)

    def load_pretrained_network(self):
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
            new_state_dict = {}
            for key in state_dict.keys():
                new_state_dict[key.replace("model.", "")] = state_dict[key]
        self.model.load_state_dict(new_state_dict)

    def forward(self, input_imgs):
        input_imgs = input_imgs.to(self.opts["gpu_id"]).float()
        if self.opts["texture"]:
            pred_codes, self.textures = self.model.forward(input_imgs)
        else:
            pred_codes, _ = self.model.forward(input_imgs)

        (
            shape__pred,
            scale_pred,
            trans_pred,
            pose_pred,
            betas_pred,
            betas_scale_pred,
        ) = pred_codes

        # TODO: rewrite to get_smal_verts()? or no? add delta_v param to self.smal()
        if self.opts["pipeline_version"] == "smalst":  # Silvia
            verts = self.model.get_smal_verts(
                pose_pred, betas_pred, trans_pred, del_v=None
            )
            faces = self.smal.faces.view(1, -1, 3)
            cam_center = torch.Tensor(
                [input_imgs.shape[2] // 2, input_imgs.shape[3] // 2]
            ).cuda(device=self.opts["gpu_id"])
            if scale_pred.shape[0] == 1:
                cam_pred = torch.cat([scale_pred, cam_center[None, :]], 1)
            else:
                cam_pred = torch.cat(
                    [
                        scale_pred.permute(1, 0),
                        cam_center.repeat(scale_pred.shape[0], 1).permute(1, 0),
                    ]
                ).permute(1, 0)
        else:  # Benjamin
            verts, joints, _ = self.smal(
                betas_pred, pose_pred, trans=trans_pred, betas_logscale=betas_scale_pred
            )
            faces = self.smal.faces.unsqueeze(0).expand(verts.shape[0], 7774, 3)

            cam_pred = torch.cat(
                [
                    scale_pred[:, [0]],
                    torch.ones(self.opts["batch_size"], 2).cuda()
                    * self.opts["img_size"]
                    / 2,
                ],
                dim=1,
            )

        # Render the prediction
        if self.vert2kp is not None:
            kp_verts = torch.matmul(self.vert2kp, verts)
        else:
            kp_verts = joints[:, self.joint2kp]
        kp_pred = self.renderer.project_points(kp_verts, cam_pred)

        _, mask_pred = self.renderer(verts, faces, cam_pred, self.opts["gpu_id"])
        # synth_rgb = torch.clamp(synth_rgb, 0.0, 1.0)
        mask_pred = mask_pred.unsqueeze(1)

        if self.opts["use_scaling_betas"]:
            betas_pred = torch.cat([betas_pred, betas_scale_pred], dim=1)

        preds = {
            "pose_pred": pose_pred,
            "betas_pred": betas_pred,
            "camera_pred": cam_pred,
            "trans_pred": trans_pred,
            "verts_pred": verts,
            # "joints_pred": joints,
            # "faces_pred": faces,
            # "kp_verts_pred": kp_verts,
            "kp_pred": kp_pred,
            "mask_pred": mask_pred,
        }
        return preds