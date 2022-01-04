"""
Mesh net model.

Original author: Silvia Zuffi (https://github.com/silviazuffi/smalst)
Edits from: Benjamin Biggs (https://github.com/benjiebob/WLDO)
Refactoring and Python 3 porting: Valeria Iegorova (https://github.com/iegorval/smalmv)
MIT licence
"""
import os
from unicodedata import bidirectional
import numpy as np
import pickle as pkl
from numpy.core.fromnumeric import shape
import torch

torch.cuda.empty_cache()
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from smal_model.smal_basics import load_smal_model
from smal_model.smal_torch import SMAL

from utils import mesh
from utils import geometry as geom_utils
from src import net_blocks as nb

# ------------- Modules ------------#
# ----------------------------------#
class ResNetConv(nn.Module):
    def __init__(self, n_blocks=4, opts=None):
        super(ResNetConv, self).__init__()
        if opts["use_resnet50"]:
            self.resnet = torchvision.models.resnet50(pretrained=True)
        else:
            self.resnet = torchvision.models.resnet18(pretrained=True)
        self.n_blocks = n_blocks
        self.opts = opts
        if self.opts["use_double_input"]:
            self.fc = nb.fc_stack(512 * 16 * 8, 512 * 8 * 8, 2)

    def forward(self, x, y=None):
        if self.opts["use_double_input"] and y is not None:
            x = torch.cat([x, y], 2)
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
        if self.opts["use_double_input"] and y is not None:
            x = x.view(x.size(0), -1)
            x = self.fc.forward(x)
            x = x.view(x.size(0), 512, 8, 8)

        return x, y


class Encoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, opts):
        super(Encoder, self).__init__()
        self.opts = opts
        self.resnet_conv = ResNetConv(n_blocks=4, opts=opts)
        num_norm_groups = opts["bottleneck_size"] // opts["channels_per_group"]
        if opts["use_resnet50"]:
            self.enc_conv1 = nb.conv2d(
                "group",
                2048,
                opts["bottleneck_size"],
                stride=2,
                kernel_size=4,
                num_groups=num_norm_groups,
            )
        else:
            self.enc_conv1 = nb.conv2d(
                "group",
                512,
                opts["bottleneck_size"],
                stride=2,
                kernel_size=4,
                num_groups=num_norm_groups,
            )

        nc_input = (
            opts["bottleneck_size"]
            * (opts["img_size"] // 64)
            * (opts["img_size"] // 64)
        )
        self.enc_fc = nb.fc_stack(nc_input, opts["nz_feat"], 2, "batch")
        self.nenc_feat = nc_input
        nb.net_init(self.enc_conv1)

    def forward(self, img, fg_img):
        # Benjamin
        resnet_feat, feat_afterconv1 = self.resnet_conv.forward(img, fg_img)
        out_enc_conv1 = self.enc_conv1(resnet_feat)
        out_enc_conv1 = out_enc_conv1.view(img.size(0), -1)
        feat = self.enc_fc.forward(out_enc_conv1)
        return feat, out_enc_conv1, feat_afterconv1


class TexturePredictorUV(nn.Module):
    """
    Outputs mesh texture
    """

    def __init__(
        self,
        uv_sampler,
        opts,
        img_H=64,
        img_W=128,
        n_upconv=5,
        nc_init=256,
        predict_flow=False,
        num_sym_faces=624,
        tex_masks=None,
        vt=None,
        ft=None,
    ):
        super(TexturePredictorUV, self).__init__()
        self.opts = opts
        self.feat_H = img_H // (2 ** n_upconv)
        self.feat_W = img_W // (2 ** n_upconv)
        self.nc_init = nc_init
        self.num_sym_faces = num_sym_faces
        self.F = uv_sampler.size(1)
        self.T = uv_sampler.size(2)
        self.predict_flow = predict_flow
        self.tex_masks = tex_masks

        # Convert texture masks into the nmr format
        # B x F x T x T x 2 --> B x F x T*T x 2
        self.uv_sampler = uv_sampler.view(-1, self.F, self.T * self.T, 2)

        if opts["number_of_textures"] > 0:
            self.enc = nn.ModuleList(
                [
                    nb.fc_stack(
                        opts["nz_feat"],
                        self.nc_init * self.feat_H[i] * self.feat_W[i],
                        2,
                        "batch",
                    )
                    for i in range(opts["number_of_textures"])
                ]
            )
        else:
            self.enc = nb.fc_stack(
                opts["nz_feat"], self.nc_init * self.feat_H * self.feat_W, 2, "batch"
            )

        if predict_flow:
            nc_final = 2
        else:
            nc_final = 3
        if opts["number_of_textures"] > 0:
            num_groups = nc_init // opts["channels_per_group"]
            self.decoder = nn.ModuleList(
                [
                    nb.decoder2d(
                        n_upconv,
                        None,
                        nc_init,
                        norm_type="group",
                        num_groups=num_groups,
                        init_fc=False,
                        nc_final=nc_final,
                        use_deconv=opts["use_deconv"],
                        upconv_mode=opts["upconv_mode"],
                    )
                    for _ in range(opts["number_of_textures"])
                ]
            )
            self.uvimage_pred_layer = [None] * opts["number_of_textures"]
        else:
            num_groups = nc_init // opts["channels_per_group"]
            self.decoder = nb.decoder2d(
                n_upconv,
                None,
                nc_init,
                norm_type="group",
                num_groups=num_groups,
                init_fc=False,
                nc_final=nc_final,
                use_deconv=opts["use_deconv"],
                upconv_mode=opts["upconv_mode"],
            )

    def forward(self, feat):
        if self.opts["number_of_textures"] > 0:
            tex_pred_layer = [None] * self.opts["number_of_textures"]
            uvimage_pred_layer = [None] * self.opts["number_of_textures"]
            for i in range(self.opts["number_of_textures"]):
                uvimage_pred_layer[i] = self.enc[i].forward(feat)
                uvimage_pred_layer[i] = uvimage_pred_layer[i].view(
                    uvimage_pred_layer[i].size(0),
                    self.nc_init,
                    self.feat_H[i],
                    self.feat_W[i],
                )
                # B x 2 or 3 x H x W
                self.uvimage_pred_layer[i] = self.decoder[i].forward(
                    uvimage_pred_layer[i]
                )
                self.uvimage_pred_layer[i] = torch.tanh(self.uvimage_pred_layer[i])

            # Compose the predicted texture maps
            # Composition by tiling
            if self.opts["number_of_textures"] == 7:
                upper = torch.cat(
                    (
                        uvimage_pred_layer[0],
                        uvimage_pred_layer[1],
                        uvimage_pred_layer[2],
                    ),
                    3,
                )
                lower = torch.cat((uvimage_pred_layer[4], uvimage_pred_layer[5]), 3)
                right = torch.cat((uvimage_pred_layer[3], uvimage_pred_layer[6]), 2)
                uvimage_pred = torch.cat((torch.cat((upper, lower), 2), right), 3)

                upper = torch.cat(
                    (
                        self.uvimage_pred_layer[0],
                        self.uvimage_pred_layer[1],
                        self.uvimage_pred_layer[2],
                    ),
                    3,
                )
                lower = torch.cat(
                    (self.uvimage_pred_layer[4], self.uvimage_pred_layer[5]), 3
                )
                right = torch.cat(
                    (self.uvimage_pred_layer[3], self.uvimage_pred_layer[6]), 2
                )
                self.uvimage_pred = torch.cat((torch.cat((upper, lower), 2), right), 3)
            elif self.opts["number_of_textures"] == 4:
                uvimage_pred = torch.cat(
                    (
                        torch.cat(
                            (
                                uvimage_pred_layer[0],
                                torch.cat(
                                    (uvimage_pred_layer[1], uvimage_pred_layer[2]), 3
                                ),
                            ),
                            2,
                        ),
                        uvimage_pred_layer[3],
                    ),
                    3,
                )
                self.uvimage_pred = torch.cat(
                    (
                        torch.cat(
                            (
                                self.uvimage_pred_layer[0],
                                torch.cat(
                                    (
                                        self.uvimage_pred_layer[1],
                                        self.uvimage_pred_layer[2],
                                    ),
                                    3,
                                ),
                            ),
                            2,
                        ),
                        self.uvimage_pred_layer[3],
                    ),
                    3,
                )
        else:
            uvimage_pred = self.enc.forward(feat)
            uvimage_pred = uvimage_pred.view(
                uvimage_pred.size(0), self.nc_init, self.feat_H, self.feat_W
            )
            # B x 2 or 3 x H x W
            self.uvimage_pred = self.decoder.forward(uvimage_pred)
            self.uvimage_pred = torch.nn.functional.tanh(self.uvimage_pred)

        tex_pred = torch.nn.functional.grid_sample(
            self.uvimage_pred, self.uv_sampler, align_corners=True
        )
        tex_pred = tex_pred.view(
            self.uvimage_pred.size(0), -1, self.F, self.T, self.T
        ).permute(0, 2, 3, 4, 1)

        if self.opts["symmetric_texture"]:
            # Symmetrize.
            tex_left = tex_pred[:, -self.num_sym_faces :]
            return torch.cat([tex_pred, tex_left], 1)
        else:
            # Contiguous Needed after the permute..
            return tex_pred.contiguous()


class ShapePredictor(nn.Module):
    """
    Outputs mesh deformations
    """

    def __init__(self, nz_feat, num_verts, opts, left_idx, right_idx, shapedirs):
        super(ShapePredictor, self).__init__()
        self.opts = opts
        if opts["use_delta_v"]:
            if opts["use_sym_idx"]:
                self.left_idx = left_idx
                self.right_idx = right_idx
                self.num_verts = num_verts
                B = shapedirs.reshape([shapedirs.shape[0], num_verts, 3])[:, left_idx]
                B = B.reshape([B.shape[0], -1])
                self.pred_layer = nn.Linear(nz_feat, len(left_idx) * 3)
            else:
                B = shapedirs
                self.pred_layer = nn.Linear(nz_feat, num_verts * 3)

            if opts["use_smal_betas"]:
                # Initialize pred_layer weights to be small so initial def aren't so big
                self.pred_layer.weight.data.normal_(0, 0.0001)
            else:
                self.fc = nb.fc("batch", nz_feat, opts["n_shape_feat"])
                n_feat = opts["n_shape_feat"]
                B = B.permute(1, 0)
                A = torch.Tensor(np.zeros((B.size(0), n_feat)))
                n = np.min((B.size(1), n_feat))
                A[:, :n] = B[:, :n]
                self.pred_layer.weight.data = torch.nn.Parameter(A)
                self.pred_layer.bias.data.fill_(0.0)

        else:
            self.ref_delta_v = torch.Tensor(
                np.zeros((opts["batch_size"], num_verts, 3))
            ).cuda(device=opts["gpu_id"])

    def forward(self, feat):
        if self.opts["use_sym_idx"]:
            delta_v = torch.Tensor(
                np.zeros((self.opts["batch_size"], self.num_verts, 3))
            ).cuda(device=self.opts["gpu_id"])
            feat = self.fc(feat)
            self.shape_f = feat

            half_delta_v = self.pred_layer.forward(feat)
            half_delta_v = half_delta_v.view(half_delta_v.size(0), -1, 3)
            delta_v[:, self.left_idx, :] = half_delta_v
            half_delta_v[:, :, 1] = -1.0 * half_delta_v[:, :, 1]
            delta_v[:, self.right_idx, :] = half_delta_v
        else:
            delta_v = self.pred_layer.forward(feat)
            # Make it B x num_verts x 3
            delta_v = delta_v.view(delta_v.size(0), -1, 3)
        # print('shape: ( Mean = {}, Var = {} )'.format(delta_v.mean().data[0], delta_v.var().data[0]))
        return delta_v


class PosePredictor(nn.Module):
    """"""

    def __init__(self, opts, num_joints=35):
        super(PosePredictor, self).__init__()
        self.opts = opts
        self.num_joints = num_joints
        self.pred_layer = nn.Linear(self.opts["nz_feat"], num_joints * 3)
        # Benjamin
        self.pred_layer.weight.data.normal_(0, 1e-4)
        self.pred_layer.bias.data.normal_(0, 1e-4)

    def forward(self, feat):
        pose = self.opts["pose_var"] * self.pred_layer.forward(feat)
        # Add this to have zero to correspond to frontal facing
        pose[:, 0] += 1.20919958
        pose[:, 1] += 1.20919958
        pose[:, 2] += -1.20919958
        return pose


class BetaScalePredictor(nn.Module):
    def __init__(self, nenc_feat, num_beta_scale=6, model_mean=None):
        super(BetaScalePredictor, self).__init__()
        self.model_mean = model_mean
        self.pred_layer = nn.Linear(nenc_feat, num_beta_scale)
        # bjb_edit
        self.pred_layer.weight.data.normal_(0, 1e-4)
        if model_mean is not None:
            self.pred_layer.bias.data = model_mean + torch.randn_like(model_mean) * 1e-4
        else:
            self.pred_layer.bias.data.normal_(0, 1e-4)

    def forward(self, feat, enc_feat):
        betas = self.pred_layer.forward(enc_feat)
        return betas


class BetasPredictor(nn.Module):
    def __init__(self, opts, nenc_feat, model_mean=None):
        super(BetasPredictor, self).__init__()
        self.opts = opts
        self.pred_layer = nn.Linear(nenc_feat, opts["num_betas"])
        # Benjamin
        self.model_mean = model_mean
        self.pred_layer.weight.data.normal_(0, 1e-4)
        if model_mean is not None:
            self.pred_layer.bias.data = model_mean + torch.randn_like(model_mean) * 1e-4
        else:
            self.pred_layer.bias.data.normal_(0, 1e-4)

    def forward(self, feat, enc_feat):
        betas = self.pred_layer.forward(enc_feat)
        return betas


class Keypoints2DPredictor(nn.Module):
    def __init__(self, opts, nz_feat, nenc_feat, num_keypoints=28):
        super(Keypoints2DPredictor, self).__init__()
        self.opts = opts
        self.num_keypoints = num_keypoints
        self.pred_layer = nn.Linear(nz_feat, 2 * num_keypoints)

    def forward(self, feat, enc_feat):
        keypoints2D = self.pred_layer.forward(feat)
        return keypoints2D.view(-1, self.num_keypoints, 2)


class ScalePredictor(nn.Module):
    """
    In case of perspective projection scale is focal length
    """

    def __init__(self, opts):
        super(ScalePredictor, self).__init__()
        self.opts = opts
        if opts["use_camera"]:
            self.opts = opts
            self.pred_layer = nn.Linear(opts["nz_feat"], opts["scale_bias"])
        else:
            scale = np.zeros((opts["batch_size"], 1))
            scale[:, 0] = 0.0
            self.ref_camera = torch.Tensor(scale).cuda(device=opts["gpu_id"])

    def forward(self, feat):
        if not self.opts["use_camera"]:
            return self.ref_camera
        if self.opts["norm_f0"] != 0:
            off = 0.0
        else:
            off = 1.0
        scale = self.pred_layer.forward(feat) + off
        return scale


class TransPredictor(nn.Module):
    """
    Outputs [tx, ty] or [tx, ty, tz]
    """

    def __init__(self, opts):
        super(TransPredictor, self).__init__()
        self.opts = opts
        if self.opts["projection_type"] == "orth":
            self.pred_layer = nn.Linear(opts["nz_feat"], 2)
        elif self.opts["projection_type"] == "perspective":
            self.pred_layer_xy = nn.Linear(opts["nz_feat"], 2)
            self.pred_layer_z = nn.Linear(opts["nz_feat"], 1)
            self.pred_layer_xy.weight.data.normal_(0, 0.0001)
            self.pred_layer_xy.bias.data.normal_(0, 0.0001)
            self.pred_layer_z.weight.data.normal_(0, 0.0001)
            self.pred_layer_z.bias.data.normal_(0, 0.0001)
        else:
            print("Unknown projection type")

    def forward(self, feat):
        trans = torch.Tensor(np.zeros((feat.shape[0], 3))).cuda(
            device=self.opts["gpu_id"]
        )
        f = torch.Tensor(np.zeros((feat.shape[0], 1))).cuda(device=self.opts["gpu_id"])
        feat_xy = feat
        feat_z = feat
        trans[:, :2] = self.pred_layer_xy(feat_xy)
        trans[:, 0] += 1.0
        trans[:, 2] = 1.0 + self.pred_layer_z(feat_z)[:, 0]

        if self.opts["fix_trans"]:
            trans[:, 2] = 1.0

        # print('trans: ( Mean = {}, Var = {} )'.format(trans.mean().data[0], trans.var().data[0]))
        return trans


class CodePredictor(nn.Module):
    def __init__(
        self,
        nenc_feat=2048,
        num_verts=1000,
        opts=None,
        left_idx=None,
        right_idx=None,
        shapedirs=None,  # TODO: WHAT IS SHAPEDIRS?
        shape_init=None,  # bjb edit TODO: REQUIRED?
    ):
        super(CodePredictor, self).__init__()
        self.opts = opts
        if opts["use_shape_predictor"]:
            self.shape_predictor = ShapePredictor(
                opts["nz_feat"],
                num_verts=num_verts,
                opts=self.opts,
                left_idx=left_idx,
                right_idx=right_idx,
                shapedirs=shapedirs,
            )
        self.scale_predictor = ScalePredictor(self.opts)
        self.trans_predictor = TransPredictor(self.opts)
        if opts["use_smal_pose"]:
            self.pose_predictor = PosePredictor(self.opts)
        if opts["use_smal_betas"]:
            scale_init = None
            if shape_init is not None:
                shape_init = shape_init[:20]
                if shape_init.shape[0] == 26:
                    scale_init = shape_init[20:]
            self.betas_predictor = BetasPredictor(
                self.opts,
                nenc_feat,
                model_mean=shape_init,
            )
            if opts["use_scaling_betas"]:
                self.betas_scale_predictor = BetaScalePredictor(
                    nenc_feat, num_beta_scale=6, model_mean=scale_init
                )

    def forward(self, feat, enc_feat):
        if self.opts["use_shape_predictor"]:
            if self.opts["use_delta_v"]:
                shape_pred = self.shape_predictor.forward(feat)
            else:
                shape_pred = self.shape_predictor.ref_delta_v
        else:
            shape_pred = None
        if self.opts["use_camera"]:
            scale_pred = self.scale_predictor.forward(feat)
        else:
            scale_pred = self.scale_predictor.ref_camera

        trans_pred = self.trans_predictor.forward(feat)

        if self.opts["use_smal_pose"]:
            pose_pred = self.pose_predictor.forward(feat)
        else:
            pose_pred = None

        if self.opts["use_smal_betas"]:
            betas_pred = self.betas_predictor.forward(feat, enc_feat)
        else:
            betas_pred = None

        if self.opts["use_scaling_betas"]:
            betas_scale_pred = self.betas_scale_predictor.forward(feat, enc_feat)[:, :6]
        else:
            betas_scale_pred = None

        return (
            shape_pred,
            scale_pred,
            trans_pred,
            pose_pred,
            betas_pred,
            betas_scale_pred,
        )


class TemporalEncoder(nn.Module):
    def __init__(self, opts, input_size):
        super(TemporalEncoder, self).__init__()
        self.opts = opts

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=opts["hidden_size_rnn"],
            bidirectional=opts["bidirectional_rnn"],
            num_layers=opts["num_layers_rnn"],
            batch_first=True,
        )
        if opts["bidirectional_rnn"]:
            linear = nn.Linear(opts["hidden_size_rnn"] * 2, opts["nz_feat"])
            linear_inverse = nn.Linear(opts["hidden_size_rnn"] * 2, input_size)
        else:
            linear = nn.Linear(opts["hidden_size_rnn"], opts["nz_feat"])
            linear_inverse = nn.Linear(opts["hidden_size_rnn"], input_size)
        self.fc = nn.Sequential(nn.ReLU(), linear, nn.Flatten(start_dim=0, end_dim=1))
        nb.net_init(self.fc)
        self.fc_inverse = nn.Sequential(
            nn.ReLU(), linear_inverse, nn.Flatten(start_dim=0, end_dim=1)
        )
        nb.net_init(self.fc_inverse)

    def forward(self, enc_feat_window):
        hidden_rnn, _ = self.gru(enc_feat_window)
        feat_rnn = self.fc(hidden_rnn)
        enc_feat_rnn = self.fc_inverse(hidden_rnn)
        # print("!!!", feat_rnn.shape, enc_feat_window.shape)
        feat_rnn = feat_rnn.view(self.opts["batch_size"], self.opts["seq_length"], -1)
        enc_feat_rnn = enc_feat_rnn.view(
            self.opts["batch_size"], self.opts["seq_length"], -1
        )
        return feat_rnn, enc_feat_rnn


# ------------ Mesh Net - Valeria's version------------#
# ----------------------------------#
class MeshNetTemporal(nn.Module):
    def __init__(self, opts, shape_init=None):
        super(MeshNetTemporal, self).__init__()
        self.shape_init = shape_init
        self.opts = opts

        self.cnn_encoder = Encoder(self.opts)

        self.rnn_encoder = TemporalEncoder(
            self.opts, input_size=self.cnn_encoder.nenc_feat
        )

        self.code_predictor = CodePredictor(
            nenc_feat=self.cnn_encoder.nenc_feat,
            opts=opts,
            shape_init=shape_init,
        )

    def forward(self, imgs_window, enc_feat_window=None):
        imgs_window = imgs_window.squeeze().cuda(device=self.opts["gpu_id"])
        if enc_feat_window is None:
            raise "Not Implemented (passing images instead of features)"
        enc_feat_window = enc_feat_window[:, :, 0, :].cuda(device=self.opts["gpu_id"])
        feat_rnn, enc_feat_rnn = self.rnn_encoder.forward(enc_feat_window)

        preds = {"pose_pred": [], "camera_pred": [], "trans_pred": []}
        for t in range(self.opts["seq_length"]):
            pred_codes = self.code_predictor(feat_rnn[:, t, :], enc_feat_rnn[:, t, :])
            (
                _,
                scale_pred,
                trans_pred,
                pose_pred,
                betas_pred,
                betas_scale_pred,
            ) = pred_codes

            cam_pred = torch.cat(
                [
                    scale_pred[:, [0]],
                    torch.ones(self.opts["batch_size"], 2).cuda()
                    * self.opts["img_size"]
                    / 2,
                ],
                dim=1,
            )
            preds["pose_pred"].append(pose_pred)
            preds["camera_pred"].append(cam_pred)
            preds["trans_pred"].append(trans_pred)

        preds["betas_pred"] = betas_pred
        if self.opts["use_scaling_betas"]:
            preds["betas_scale_pred"] = betas_scale_pred
        return preds


# ------------ Mesh Net - Benjamin's version------------#
# ----------------------------------#
class MeshNet(nn.Module):
    def __init__(self, opts, shape_init=None):
        super(MeshNet, self).__init__()
        self.shape_init = shape_init
        self.opts = opts

        self.encoder = Encoder(self.opts)

        self.code_predictor = CodePredictor(
            nenc_feat=self.encoder.nenc_feat,
            opts=opts,
            shape_init=shape_init,
        )

    def forward(self, img, masks=None):
        img_feat, enc_feat, enc_feat_afterconv1 = self.encoder.forward(img, masks)
        codes_pred = self.code_predictor.forward(img_feat, enc_feat)
        return codes_pred, enc_feat_afterconv1


# ------------ Mesh Net - Silvia's version------------#
# ----------------------------------#
class MeshNetSMALST(nn.Module):
    def __init__(
        self,
        opts,
        sfm_mean_shape=None,
        tex_masks=None,
    ):
        # Input shape is H x W of the image.
        super(MeshNetSMALST, self).__init__()
        self.opts = opts
        self.tex_masks = tex_masks

        # Instantiate the SMAL model in Torch
        # model_path = os.path.join(self.opts["model_dir"], self.opts["model_name"])
        self.smal = SMAL(opts=self.opts)  # TODO: move to smal_predictor as in SMBLD

        self.left_idx = np.hstack((self.smal.left_inds, self.smal.center_inds))
        self.right_idx = np.hstack((self.smal.right_inds, self.smal.center_inds))

        pose = np.zeros((1, 105))
        betas = np.zeros((1, self.opts["num_betas"]))
        V, _, _ = self.smal(
            torch.Tensor(betas).cuda(device=self.opts["gpu_id"]),
            torch.Tensor(pose).cuda(device=self.opts["gpu_id"]),
        )
        verts = V[0, :, :]
        verts = verts.data.cpu().numpy()
        faces = self.smal.f

        num_verts = verts.shape[0]

        if self.opts["symmetric"]:
            (
                verts,
                faces,
                num_indept,
                num_sym,
                num_indept_faces,
                num_sym_faces,
            ) = mesh.make_symmetric(
                verts,
                faces,
                self.smal.left_inds,
                self.smal.right_inds,
                self.smal.center_inds,
            )
            if sfm_mean_shape is not None:
                verts = geom_utils.project_verts_on_mesh(
                    verts, sfm_mean_shape[0], sfm_mean_shape[1]
                )

            num_sym_output = num_indept + num_sym
            if opts["only_mean_sym"]:
                print("Only the mean shape is symmetric!")
                self.num_output = num_verts
            else:
                self.num_output = num_sym_output
            self.num_sym = num_sym
            self.num_indept = num_indept
            self.num_indept_faces = num_indept_faces
            self.num_sym_faces = num_sym_faces
            # mean shape is only half.
            self.mean_v = nn.Parameter(torch.Tensor(verts[:num_sym_output]))

            # Needed for symmetrizing..
            self.flip = Variable(
                torch.ones(1, 3).cuda(device=self.opts["gpu_id"]), requires_grad=False
            )
            self.flip[0, 0] = -1
        else:
            if sfm_mean_shape is not None:
                verts = geom_utils.project_verts_on_mesh(
                    verts, sfm_mean_shape[0], sfm_mean_shape[1]
                )
            self.mean_v = nn.Parameter(torch.Tensor(verts))
            self.num_output = num_verts
            faces = faces.astype(np.int32)

        verts_np = verts
        faces_np = faces
        self.faces = Variable(
            torch.LongTensor(faces).cuda(device=self.opts["gpu_id"]),
            requires_grad=False,
        )
        self.edges2verts = mesh.compute_edges2verts(verts, faces)

        vert2kp_init = torch.Tensor(
            np.ones((opts["num_kps"], num_verts)) / float(num_verts)
        )
        # Remember initial vert2kp (after softmax)
        self.vert2kp_init = torch.nn.functional.softmax(
            Variable(
                vert2kp_init.cuda(device=self.opts["gpu_id"]), requires_grad=False
            ),
            dim=1,
        )
        self.vert2kp = nn.Parameter(vert2kp_init)

        self.encoder = Encoder(self.opts)

        self.code_predictor = CodePredictor(
            nenc_feat=self.encoder.nenc_feat,
            opts=opts,
            num_verts=self.num_output,
            left_idx=self.left_idx,
            right_idx=self.right_idx,
            shapedirs=self.smal.shapedirs,
        )

        if self.opts["texture"]:
            if self.opts["symmetric_texture"]:
                num_faces = self.num_indept_faces + self.num_sym_faces
            else:
                num_faces = faces.shape[0]
                self.num_sym_faces = 0

            # Instead of loading an obj file
            uv_data = pkl.load(
                open(os.path.join(self.opts["model_dir"], opts["uv_data_file"]), "rb"),
                encoding="latin1",
            )
            vt = uv_data["vt"]
            ft = uv_data["ft"]
            self.vt = vt
            self.ft = ft
            uv_sampler = mesh.compute_uvsampler(
                verts_np, faces_np[:num_faces], vt, ft, tex_size=opts["tex_size"]
            )
            # F' x T x T x 2
            uv_sampler = Variable(
                torch.FloatTensor(uv_sampler).cuda(device=self.opts["gpu_id"]),
                requires_grad=False,
            )
            # B x F' x T x T x 2
            uv_sampler = uv_sampler.unsqueeze(0).repeat(
                self.opts["batch_size"], 1, 1, 1, 1
            )
            if opts["number_of_textures"] > 0:
                if opts["texture_img_size"] == 256:
                    if opts["number_of_textures"] == 7:
                        img_H = np.array([96, 96, 96, 96, 160, 160, 160])
                        img_W = np.array([64, 128, 32, 32, 128, 96, 32])
                    elif opts["number_of_textures"] == 4:
                        img_H = np.array([96, 160, 160, 256])
                        img_W = np.array([224, 128, 96, 32])
                else:
                    raise "Wrong Texture Size (check opts['number_of_textures'])"
            else:
                img_H = opts["texture_img_size"]
                img_W = opts["texture_img_size"]

            self.texture_predictor = TexturePredictorUV(
                uv_sampler,
                opts,
                img_H=img_H,
                img_W=img_W,
                predict_flow=True,
                num_sym_faces=self.num_sym_faces,
                tex_masks=self.tex_masks,
                # vt=vt,
                # ft=ft,
            )

            nb.net_init(self.texture_predictor)

    def forward(self, img, masks=None):
        if self.opts["is_optimization"]:
            if self.opts["is_var_opt"]:
                img_feat, enc_feat, enc_feat_afterconv1 = self.encoder.forward(
                    img, masks
                )
                codes_pred = self.code_predictor.forward(img_feat, enc_feat)
                self.opts_scale = Variable(
                    codes_pred[1].cuda(device=self.opts["gpu_id"]), requires_grad=True
                )
                self.opts_pose = Variable(
                    codes_pred[3].cuda(device=self.opts["gpu_id"]), requires_grad=True
                )
                self.opts_trans = Variable(
                    codes_pred[2].cuda(device=self.opts["gpu_id"]), requires_grad=True
                )
                self.opts_delta_v = Variable(
                    codes_pred[0].cuda(device=self.opts["gpu_id"]), requires_grad=True
                )
                self.op_features = [
                    self.opts_scale,
                    self.opts_pose,
                    self.opts_trans,
                ]
                codes_pred = (
                    self.opts_delta_v,
                    self.opts_scale,
                    self.opts_trans,
                    self.opts_pose,
                    None,
                    None,
                )
            else:
                img_feat, enc_feat, enc_feat_afterconv1 = self.encoder.forward(
                    img, masks
                )
                self.op_features = Variable(
                    img_feat.cuda(device=self.opts["gpu_id"]), requires_grad=True
                )
                codes_pred = self.code_predictor.forward(self.op_features, None)
                img_feat = self.op_features
        else:
            img_feat, enc_feat, enc_feat_afterconv1 = self.encoder.forward(img, masks)
            codes_pred = self.code_predictor.forward(img_feat, enc_feat)
        if self.opts["texture"]:
            texture_pred = self.texture_predictor.forward(img_feat)
            return codes_pred, texture_pred
        else:
            return codes_pred, enc_feat_afterconv1

    # def symmetrize(self, V):
    #     """
    #     Takes num_indept+num_sym verts and makes it
    #     num_indept + num_sym + num_sym
    #     Is identity if model is not symmetric
    #     """
    #     if self.opts["symmetric"]:
    #         if V.dim() == 2:
    #             # No batch
    #             V_left = self.flip * V[-self.num_sym :]
    #             return torch.cat([V, V_left], 0)
    #         else:
    #             # With batch
    #             V_left = self.flip * V[:, -self.num_sym :]
    #             return torch.cat([V, V_left], 1)
    #     else:
    #         return V

    def get_smal_verts(self, pose=None, betas=None, trans=None, del_v=None):
        if pose is None:
            pose = torch.Tensor(np.zeros((1, 105))).cuda(device=self.opts["gpu_id"])
        if betas is None:
            betas = torch.Tensor(np.zeros((1, self.opts["num_betas"]))).cuda(
                device=self.opts["gpu_id"]
            )
        if trans is None:
            trans = torch.Tensor(np.zeros((1, 3))).cuda(device=self.opts["gpu_id"])

        verts, _, _ = self.smal(betas, pose, trans, del_v)
        return verts

    # def get_mean_shape(self):
    #     return self.symmetrize(self.mean_v)