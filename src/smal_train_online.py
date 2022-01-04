import os
import json
import shutil
import imageio
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
from data.smal_online_loader import data_loader
from src import smal_mesh_net as mesh_net
from src.nmr import NeuralRenderer
from src import loss_utils
from utils import smal_vis
from src.animal_shape_prior import MultiShapePrior
from src.pose_prior import PosePrior
from smal_model.smal_torch import SMAL
from torch.autograd import Variable

CONFIG_DIR = "configs/"
CONFIG_DATA = "cows_data_temporal.json"
CONFIG_MODEL = "train_params_cows_temporal.json"


class TestNetwork(torch.nn.Module):
    def __init__(self, opts):
        super(TestNetwork, self).__init__()
        self.opts = opts

        self.vert2kp = torch.Tensor(
            pkl.load(open(opts["verts2kp_path"], "rb"), encoding="latin1")
        ).cuda(device=opts["gpu_id"])

        print("Setting up pipeline...")
        if opts["pipeline_version"] == "smal4v":
            self.model = mesh_net.MeshNetTemporal(opts)
        else:
            raise "Unsupported Pipeline"

        self.model = self.model.cuda(device=self.opts["gpu_id"])

        self.smal = SMAL(opts)
        faces = self.smal.faces.view(1, -1, 3)

        self.renderer = NeuralRenderer(
            opts["img_size"],
            opts["projection_type"],
            opts["norm_f"],
            opts["norm_z"],
            opts["norm_f0"],
        )
        self.renderer.directional_light_only()

        # TODO: add if eval
        self.faces = faces.repeat(opts["batch_size"], 1, 1)
        self.vis_renderer = smal_vis.VisRenderer(
            opts["img_size"],
            faces.data.cpu().numpy(),
            opts["projection_type"],
            opts["norm_f"],
            opts["norm_z"],
            opts["norm_f0"],
        )
        self.vis_renderer.set_bgcolor([1.0, 1.0, 1.0])

    def load_pretrained_single_frame(self):
        pretrained_path = os.path.join(
            self.opts["pretrained_dir"], self.opts["pretrained_single_frame_name"]
        )
        state_dict = torch.load(pretrained_path)
        encoder_dict, code_pred_dict = {}, {}
        for key in state_dict.keys():
            if "encoder" in key:
                encoder_dict[key.replace("model.module.encoder.", "")] = state_dict[key]
            if "code_predictor" in key:
                code_pred_dict[
                    key.replace("model.module.code_predictor.", "")
                ] = state_dict[key]
        self.model.cnn_encoder.load_state_dict(encoder_dict)
        self.model.code_predictor.load_state_dict(code_pred_dict)

    def load_pretrained(self):
        pretrained_path = os.path.join(
            self.opts["pretrained_dir"], self.opts["pretrained_name"]
        )
        state_dict = torch.load(pretrained_path)
        new_state_dict = {}
        for key in state_dict.keys():
            new_state_dict[key.replace("model.", "")] = state_dict[key]
        self.model.load_state_dict(new_state_dict)

    def forward(self, input_window):
        preds = self.model.forward(input_window["images"], input_window["enc_features"])
        preds["verts_pred"] = []
        preds["kp_pred"] = []
        preds["mask_pred"] = []
        betas_scale_pred = preds["betas_scale_pred"] if "betas_scale" in preds else None
        for i in range(len(preds["pose_pred"])):
            verts, _, _ = self.smal(
                preds["betas_pred"],
                preds["pose_pred"][i],
                trans=preds["trans_pred"][i],
                betas_logscale=betas_scale_pred,
            )
            faces = self.smal.faces.unsqueeze(0).expand(verts.shape[0], 7774, 3)
            # Render the predictions
            kp_verts = torch.matmul(self.vert2kp, verts)
            kp_pred = self.renderer.project_points(kp_verts, preds["camera_pred"][i])
            _, mask_pred = self.renderer(
                verts, faces, preds["camera_pred"][i], self.opts["gpu_id"]
            )
            mask_pred = mask_pred.unsqueeze(1)
            preds["verts_pred"].append(verts)
            preds["kp_pred"].append(kp_pred)
            preds["mask_pred"].append(mask_pred)
        return preds


class OnlineTrainer:
    def __init__(self, opts):
        self.opts = opts

    def init_sequence(self, seq_name):
        if self.opts["precompute_features"]:
            cnn_encoder = self.model.model.cnn_encoder
        else:
            cnn_encoder = None
        self.dataloader = data_loader(self.opts, seq_name, cnn_encoder)

    def define_criterion(self):
        self.kp_loss_fn = loss_utils.kp_l1_loss
        self.kp_temporal_fn = torch.nn.L1Loss(reduction="none")
        self.mask_loss_fn = torch.nn.L1Loss(reduction="none")
        # self.cam_temporal_fn = loss_utils.camera_loss
        # self.pose_temporal_fn = loss_utils.model_pose_loss
        self.pose_prior = PosePrior(self.opts["walking_prior"])
        data_path = os.path.join(self.opts["model_dir"], self.opts["data_name"])
        self.shape_prior = MultiShapePrior(family_name="cow", data_path=data_path)

    def init_training(self, seq_name):
        if os.path.exists(self.opts["checkpoint_dir"]):
            shutil.rmtree(self.opts["checkpoint_dir"])
        os.makedirs(self.opts["checkpoint_dir"])

        self.model = TestNetwork(self.opts).cuda(device=self.opts["gpu_id"])
        if self.opts["use_pretrained_single_frame"]:
            self.model.load_pretrained_single_frame()
        self.define_criterion()
        self.init_sequence(seq_name)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.opts["learning_rate"],
            betas=(self.opts["beta1"], 0.99),  # TODO: 0.99 as parameter
        )

    def set_input_batch(self, batch):
        self.batch_size = batch["images"].shape[0]
        self.images = batch["images"]
        self.kps = Variable(
            batch["keypoints"].type(torch.FloatTensor).cuda(device=self.opts["gpu_id"]),
            requires_grad=False,
        )
        self.masks = Variable(
            batch["masks"].type(torch.FloatTensor).cuda(device=self.opts["gpu_id"]),
            requires_grad=False,
        )

    def set_batch_loss(self, epoch):
        unbind = lambda tensor: tensor.permute(1, 0, 2, 3).flatten(0, 1).unbind(0)
        # masks_filter = torch.Tensor(
        #     [torch.count_nonzero(mask) > 0 for mask in unbind(self.masks)]
        # )
        kps_filter = torch.Tensor(
            [torch.count_nonzero(kp) > 0 for kp in unbind(self.kps)]
        )
        self.kp_loss, self.mask_loss, self.kp_temporal_loss = 0, 0, 0
        self.cam_temporal_loss, self.pose_temporal_loss = 0, 0
        self.pose_prior_loss, self.shape_prior_loss = 0, 0
        for t in range(self.opts["seq_length"]):
            # Loss with ground-truth keypoints (missing keypoints omitted)
            self.kp_loss += self.kp_loss_fn(
                self.preds["kp_pred"][t], self.kps[:, t, ...], reduction="sum"
            )
            # Loss with OSVOS masks
            # borders_vert = self.images.sum(dim=(2, 4))[:, t, :]
            # borders_hor = self.images.sum(dim=(2, 3))[:, t, :]

            # TODO rewrite without for... CHECK FOR OTHER SEQ
            mask_pred = self.preds["mask_pred"][t].squeeze()
            mask_gt = self.masks[:, t, ...].clone()
            for b in range(self.images.shape[0]):  # TODO: rewrite without for
                border_mask = (
                    torch.all(self.images[b, t, ...] < 1.0, dim=0).unsqueeze(0).float()
                ).to(device=self.opts["gpu_id"])
                mask_gt[b, ...] = (mask_gt[b, ...] * border_mask) + mask_pred[
                    b, ...
                ] * (1.0 - border_mask)

            mask_loss = self.mask_loss_fn(mask_pred, mask_gt).sum(dim=(1, 2))
            # mask_loss *= masks_filter[
            #     t * self.opts["batch_size"] : (t + 1) * self.opts["batch_size"]
            # ].cuda(device=self.opts["gpu_id"])

            self.mask_loss += torch.sum(mask_loss)
            self.pose_prior_loss += torch.mean(
                self.pose_prior(self.preds["pose_pred"][t])
            )
            # Temporal smoothing
            # if t > 0:
            #     self.cam_temporal_loss += self.cam_temporal_fn(
            #         self.preds["camera_pred"][t], self.preds["camera_pred"][t - 1]
            #     ) / (self.opts["seq_length"] - 1)
            #     self.pose_temporal_loss += self.pose_temporal_fn(
            #         self.preds["pose_pred"][t], self.preds["pose_pred"][t - 1], self.opts
            #     ) / (self.opts["seq_length"] - 1)

            kp_temporal_loss = torch.zeros_like(mask_loss)
            if t > 0:
                kp_temporal_loss += self.kp_temporal_fn(
                    self.preds["kp_pred"][t], self.preds["kp_pred"][t - 1]
                ).sum(dim=(1, 2))
            # if t < self.opts["seq_length"] - 1:
            #     kp_temporal_loss += self.kp_temporal_fn(
            #         self.preds["kp_pred"][t], self.preds["kp_pred"][t + 1]
            #     ).sum(dim=(1, 2))
            # if not (t == 0 or t == self.opts["seq_length"] - 1):
            #     kp_temporal_loss /= 2.0
            kp_temporal_loss *= 1 - kps_filter[
                t * self.opts["batch_size"] : (t + 1) * self.opts["batch_size"]
            ].cuda(device=self.opts["gpu_id"])
            self.kp_temporal_loss += torch.sum(kp_temporal_loss)

        # # self.kp_loss = (self.kp_loss + self.kp_temporal_loss) / self.opts["batch_size"]
        # keypoints loss
        if kps_filter.sum() > 0:
            self.kp_loss /= kps_filter.sum()
        if (1 - kps_filter).sum() > 0:
            self.kp_temporal_loss /= (1 - kps_filter).sum()
        self.kp_loss += self.kp_temporal_loss

        # prior losses
        self.pose_prior_loss /= self.opts["seq_length"]
        self.shape_prior_loss = torch.mean(self.shape_prior(self.preds["betas_pred"]))
        self.mask_loss /= self.opts["seq_length"] * self.opts["batch_size"]
        # silhouette loss
        if epoch > self.opts["num_epochs_only_kp"]:
            # if masks_filter.sum() > 0:
            #     self.mask_loss /= masks_filter.sum()
            self.pose_prior_loss /= 2.0
            self.shape_prior_loss /= 10.0
        else:
            self.mask_loss = 0.0

        # total batch loss
        self.batch_loss = (
            self.opts["kp_loss_wt"] * self.kp_loss
            # + self.opts["kp_temporal_wt"] * self.kp_temporal_loss
            # + self.opts["mask_loss_wt"] * self.mask_loss
            # + 1000000.0 * self.pose_temporal_loss
            + self.opts["pose_prior_wt"] * self.pose_prior_loss
            + self.opts["shape_prior_wt"] * self.shape_prior_loss
        )
        self.batch_loss += self.opts["mask_loss_wt"] * self.mask_loss
        self.total_loss += self.batch_loss

    def train(self):
        # TODO: REMOVE (only in eval -> write evaluator class)
        # opts["evaluation_path"] = "data/results/evaluation_temporal_training"
        # opts["visualizations_dir"] = "data/results/visualizations_temporal_training"
        # if os.path.exists(self.opts["evaluation_path"]):
        #     shutil.rmtree(self.opts["evaluation_path"])
        # os.makedirs(self.opts["evaluation_path"])
        # if os.path.exists(self.opts["visualizations_dir"]):
        #     shutil.rmtree(self.opts["visualizations_dir"])
        # os.makedirs(self.opts["visualizations_dir"])
        self.model.train()
        # self.model.load_pretrained()
        for epoch in range(self.opts["num_epochs"]):
            tqdm_iterator = tqdm(
                self.dataloader, desc="Online Training", total=len(self.dataloader)
            )
            self.total_loss = 0
            print(f"Epoch {epoch + 1}/{self.opts['num_epochs']}")
            for (i, batch) in enumerate(tqdm_iterator):
                self.set_input_batch(batch)
                self.optimizer.zero_grad()
                self.preds = self.model.forward(batch)
                self.set_batch_loss(epoch)

                tqdm_iterator.desc = f"batch_loss: {self.batch_loss}, kp: {self.kp_loss}, pose_t: {self.pose_temporal_loss}, mask: {self.mask_loss}, pose_p: {self.pose_prior_loss}, shape_p: {self.shape_prior_loss}"
                tqdm_iterator.update()

                # if epoch > self.opts["num_epochs_only_kp"]:
                #     if self.mask_loss == 0:
                #         print("No mask available in batch")
                #         continue
                # else:
                #     if self.kp_loss == 0:
                #         print("No keypoints available in batch")
                #         continue

                self.batch_loss.backward()
                self.optimizer.step()

                # TODO: either remove or make optional visualization part of the model
                # self.visualize(
                #     batch["images"], self.preds, batch["keypoints"], f"epoch{epoch}_batch{i}"
                # )

            norm_constant = len(self.dataloader) * self.opts["seq_length"]
            print("Mean sequence loss:", self.total_loss / norm_constant)

            if epoch % 10 == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.opts["checkpoint_dir"], f"_online_epoch{epoch}.pth"
                    ),
                )

    # move from trainer
    def eval(self, seq_name):
        opts["evaluation_path"] = "data/results/evaluation_temporal_training"
        opts["visualizations_dir"] = "data/results/visualizations_temporal_training"
        if os.path.exists(self.opts["evaluation_path"]):
            shutil.rmtree(self.opts["evaluation_path"])
        os.makedirs(self.opts["evaluation_path"])
        if os.path.exists(self.opts["visualizations_dir"]):
            shutil.rmtree(self.opts["visualizations_dir"])
        os.makedirs(self.opts["visualizations_dir"])

        self.model = TestNetwork(self.opts).cuda(device=self.opts["gpu_id"])
        self.model.load_pretrained()
        self.model.eval()
        self.init_sequence(seq_name)
        tqdm_iterator = tqdm(
            self.dataloader, desc="Online Evaluation", total=len(self.dataloader)
        )
        for (i, window) in enumerate(tqdm_iterator):
            with torch.no_grad():
                preds = self.model.forward(window)
                self.visualize(window, preds, f"window{i}")

    # move from trainer
    def visualize(
        self,
        window,
        outputs,  #: Dict[str, torch.Tensor],
        out_name_win: str,
    ):
        for t in range(len(outputs["verts_pred"])):
            vert = outputs["verts_pred"][t][0]
            cam = outputs["camera_pred"][t][0]
            shape_pred = self.model.vis_renderer(vert, opts["gpu_id"], cams=cam)
            shape_pred = shape_pred.astype(np.uint8)
            img = (
                window["images"][0, t, ...].squeeze().cpu().detach().numpy()
            )  # TODO: remove detach
            img = np.transpose(img, (1, 2, 0))

            mask = outputs["mask_pred"][t][0, 0, ...].cpu().detach().numpy()
            out_name = f"{out_name_win}_frame{t}.png"

            Iov = (0.3 * 255 * img + 0.7 * shape_pred).astype(np.uint8)
            imageio.imwrite(
                os.path.join(opts["evaluation_path"], "shape_ov_" + out_name), Iov
            )
            imageio.imwrite(
                os.path.join(opts["evaluation_path"], "shape_" + out_name), shape_pred
            )
            mask_gt = window["masks"][0, t, ...].cpu().detach().numpy()
            nrows = 4 if mask_gt.sum() > 0 else 3
            _, ax = plt.subplots(1, nrows)
            ax[0].imshow(img)
            ax[0].set_title("input")
            ax[0].axis("off")
            ax[1].imshow(img)
            ax[1].imshow(shape_pred, alpha=0.7)
            kp_gt = window["keypoints"][0, t, :, :2]
            kp_pred = outputs["kp_pred"][t][0, ...].cpu().detach().numpy()
            vis = window["keypoints"][0, t, :, 2]
            if vis.sum() > 0:
                idx = np.where(vis)
                ax[1].scatter(kp_gt[idx, 0], kp_gt[idx, 1], c="k")
                ax[1].scatter(kp_pred[idx, 0], kp_pred[idx, 1], c="r")
            ax[1].set_title("pred mesh")
            ax[1].axis("off")
            ax[2].imshow(mask, cmap="gray")
            ax[2].axis("off")
            ax[2].set_title("pred mask")
            if nrows == 4:
                ax[3].imshow(mask_gt, cmap="gray")
                ax[3].axis("off")
                ax[3].set_title("gt mask")
            out_path = os.path.join(opts["visualizations_dir"], out_name)
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()


def main(opts):
    torch.manual_seed(0)
    np.random.seed(0)
    print("Training initialization...")
    trainer = OnlineTrainer(opts)
    # trainer.eval(SEQ_NAME)
    trainer.init_training(SEQ_NAME)
    print("Done initializing training.")
    trainer.train()


SEQ_NAME = "seq295_test"
if __name__ == "__main__":
    fname_data = os.path.join(CONFIG_DIR, CONFIG_DATA)
    with open(fname_data) as f:
        opts_data = json.load(f)
    fname_model = os.path.join(CONFIG_DIR, CONFIG_MODEL)
    with open(fname_model) as f:
        opts_model = json.load(f)
    opts = opts_data | opts_model
    main(opts)