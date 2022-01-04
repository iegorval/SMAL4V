import os
import json
import shutil
import torch
import numpy as np
import torch.nn as nn
import pickle as pkl
from tqdm import tqdm
from torch.utils.data import DataLoader
from src import loss_utils
from src import smal_mesh_net as mesh_net
from src.nmr import NeuralRenderer
from src.animal_shape_prior import MultiShapePrior
from src.pose_prior import PosePrior
from smal_model.smal_torch import SMAL
from data.dogs import StanfordExtra
from data.zebra import ZebraDataset
from data.cows import AnimalKey
from data.tigdog import TigDog
from torch.autograd import Variable


# TODO: pass as flags
CONFIG_DIR = "configs/"
CONFIG_DATA = "cows_data_single_frame.json"
CONFIG_MODEL = "train_params_cows_single_frame.json"


class ParentNetwork(torch.nn.Module):  # TODO: merge with MeshPredictor?
    def __init__(self, opts):
        super(ParentNetwork, self).__init__()
        self.opts = opts

        self.vert2kp = torch.Tensor(
            pkl.load(open(opts["verts2kp_path"], "rb"), encoding="latin1")
        ).cuda(device=opts["gpu_id"])

        print("Setting up pipeline...")
        if opts["pipeline_version"] == "smbld":
            self.model = nn.DataParallel(mesh_net.MeshNet(opts))
        else:
            raise "Unsupported Pipeline"

        if opts["use_pretrained"]:
            self.load_pretrained_network()
        self.model = self.model.cuda(device=self.opts["gpu_id"])

        self.renderer = NeuralRenderer(
            opts["img_size"],
            opts["projection_type"],
            opts["norm_f"],
            opts["norm_z"],
            opts["norm_f0"],
        )
        self.renderer.directional_light_only()

        self.smal = SMAL(opts)

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
        if self.opts["texture"]:
            pred_codes, self.textures = self.model.forward(input_imgs)
        else:
            pred_codes, _ = self.model.forward(input_imgs)

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
                torch.ones(opts["batch_size"], 2).cuda() * opts["img_size"] / 2,
            ],
            dim=1,
        )

        # TODO: rewrite to get_smal_verts()? or no? add delta_v param to self.smal()
        verts, _, _ = self.smal(
            betas_pred, pose_pred, trans=trans_pred, betas_logscale=betas_scale_pred
        )
        faces = self.smal.faces.unsqueeze(0).expand(verts.shape[0], 7774, 3)

        # Render the prediction
        # self.renderer.directional_light_only()
        kp_verts = torch.matmul(self.vert2kp, verts)
        kp_pred = self.renderer.project_points(kp_verts, cam_pred)

        synth_rgb, synth_silhouettes = self.renderer(
            verts, faces, cam_pred, self.opts["gpu_id"]
        )
        synth_rgb = torch.clamp(synth_rgb, 0.0, 1.0)
        synth_silhouettes = synth_silhouettes.unsqueeze(1)

        if self.opts["use_scaling_betas"]:
            betas_pred = torch.cat([betas_pred, betas_scale_pred], dim=1)

        preds = {
            "pose": pose_pred,
            "betas": betas_pred,
            "camera": cam_pred,
            "trans": trans_pred,
            "verts": verts,
            "faces": faces,
            "kp_verts": kp_verts,
            "kp_pred": kp_pred,
            "synth_rgb": synth_rgb,
            "synth_silhouettes": synth_silhouettes,
        }
        return preds


class ShapeTrainer:
    def __init__(self, opts):
        self.opts = opts

    def set_mask(self, batch):
        if "mask" in batch.keys():
            mask_tensor = batch["mask"].type(torch.FloatTensor)
            self.masks = Variable(
                mask_tensor.cuda(device=opts["gpu_id"]), requires_grad=False
            )
        else:
            self.masks = None

    def set_keypoints(self, batch):
        if self.opts["use_keypoints"] and "kp" in batch.keys():
            kp_tensor = batch["kp_raw"].type(torch.FloatTensor)
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

    def init_dataset(self):
        if opts["dataset"] == "stanford_extra":
            self.dataset = StanfordExtra(opts, use_augmentation=True, is_train=True)
        elif opts["dataset"] == "grevy_zebra":
            self.dataset = ZebraDataset(opts)
        elif opts["dataset"] == "tigdog":
            self.dataset = TigDog(opts, train=True)
        elif opts["dataset"] == "animalkey":
            self.dataset = AnimalKey(opts, is_train=True)
        else:
            raise "Unknown Dataset"
        self.dataloader = DataLoader(
            self.dataset, batch_size=opts["batch_size"], shuffle=True
        )

    def init_training(self):
        self.init_dataset()
        self.define_criterion()
        self.model = ParentNetwork(self.opts).cuda(device=self.opts["gpu_id"])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.opts["learning_rate"],
            betas=(self.opts["beta1"], 0.999),
        )
        if os.path.exists(self.opts["checkpoint_dir"]):
            shutil.rmtree(self.opts["checkpoint_dir"])
        os.makedirs(self.opts["checkpoint_dir"])

    def define_criterion(self):
        self.kp_loss_fn = loss_utils.kp_l1_loss
        self.mask_loss_fn = loss_utils.mask_loss
        self.pose_prior = PosePrior(self.opts["walking_prior"])
        data_path = os.path.join(self.opts["model_dir"], self.opts["data_name"])
        self.shape_prior = MultiShapePrior(family_name="cow", data_path=data_path)

    def set_batch_loss(self):
        self.kp_loss = self.kp_loss_fn(self.preds["kp_pred"], self.kps)
        self.mask_loss = self.mask_loss_fn(self.preds["synth_silhouettes"], self.masks)
        self.pose_prior_loss = torch.mean(self.pose_prior(self.preds["pose"]))
        self.shape_prior_loss = torch.mean(self.shape_prior(self.preds["betas"]))

        self.batch_loss = (
            self.opts["kp_loss_wt"] * self.kp_loss
            + self.opts["mask_loss_wt"] * self.mask_loss
            + self.opts["pose_prior_wt"] * self.pose_prior_loss
            + self.opts["shape_prior_wt"] * self.shape_prior_loss
        )

        self.total_loss += self.batch_loss

    def set_input_batch(self, batch):
        # print(batch["kp_raw"].shape, "!KP", batch["img"][0, ...].shape)
        # test = np.transpose(batch["img"][0, ...], (1, 2, 0))
        # plt.imshow(test)
        # kp = batch["kp_raw"].cpu().detach().numpy()
        # idx = np.where(kp[0, :, 2].astype(int))
        # plt.scatter(kp[0, idx, 0], kp[0, idx, 1])
        # plt.imshow(batch["mask"][0, 0, ...], alpha=0.4)
        # plt.savefig("test_tigdog.png")
        # plt.close()
        input_img_tensor = batch["img"].type(torch.FloatTensor)
        self.batch_size = batch["img"].shape[0]
        self.input_imgs = Variable(
            input_img_tensor.cuda(device=self.opts["gpu_id"]), requires_grad=False
        )
        self.img_paths = batch["img_path"]
        self.set_mask(batch)
        self.set_keypoints(batch)
        self.set_camera(batch)

    def train(self):
        self.model.train()
        for epoch in range(self.opts["num_epochs"]):
            tqdm_iterator = tqdm(
                self.dataloader, desc="Train", total=len(self.dataloader)
            )
            self.total_loss = 0
            print(f"Epoch {epoch + 1}/{self.opts['num_epochs']}")
            for batch in tqdm_iterator:
                self.set_input_batch(batch)  # set ground truth for training batch
                self.optimizer.zero_grad()
                self.preds = self.model.forward(self.input_imgs)
                self.set_batch_loss()

                tqdm_iterator.desc = f"batch_loss: {self.batch_loss}, kp_loss: {self.kp_loss}, mask_loss: {self.mask_loss}, pose_prior_loss: {self.pose_prior_loss}, shape_prior_loss: {self.shape_prior_loss}"
                tqdm_iterator.update()
                if self.kp_loss == 0.0 and self.mask_loss == 0.0:
                    print("No annotations in the whole batch")
                    continue
                self.batch_loss.backward()
                self.optimizer.step()
            print("mean dataset loss:", self.total_loss / len(self.dataloader))

            if epoch % 10 == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.opts["checkpoint_dir"], f"_epoch{epoch}.pth"),
                )

    # # TEMPORARY
    # def eval(self):
    #     self.model.eval()
    #     tqdm_iterator = tqdm(self.dataloader, desc="Train", total=len(self.dataloader))
    #     self.total_loss = 0
    #     TEST_PATH = "data/results/test_vis"
    #     os.makedirs(TEST_PATH, exist_ok=True)
    #     for batch in tqdm_iterator:
    #         with torch.no_grad():
    #             self.set_input_batch(batch)
    #             self.preds = self.model(self.input_imgs)
    #             self.preds.update(batch)
    #             self.set_batch_loss()

    #             tqdm_iterator.desc = (
    #                 f"batch_loss: {self.batch_loss}, kp_loss: {self.kp_loss}"
    #             )
    #             tqdm_iterator.update()

    #             for i in range(self.input_imgs.shape[0]):
    #                 shape_pred = self.preds["synth_rgb"][i, ...].cpu().detach().numpy()
    #                 # shape_pred = np.flipud(np.transpose(shape_pred, [1, 2, 0])) * 255
    #                 shape_pred = np.transpose(shape_pred, [1, 2, 0]) * 255
    #                 kp_pred = self.preds["kp_pred"][i, ...].cpu().detach().numpy()
    #                 kps = self.kps[i, ...].cpu().detach().numpy()
    #                 img = self.input_imgs[i, ...].squeeze().cpu().detach().numpy()
    #                 img = np.transpose(img, [1, 2, 0]) * 255
    #                 img_pred = (0.3 * img + 0.7 * shape_pred).astype(np.uint8)
    #                 img_name = self.preds["img_path"][i].split("/")[-1]
    #                 img_file = os.path.join(TEST_PATH, img_name)
    #                 fig, ax = plt.subplots()
    #                 vis = kps[:, 2].astype(bool)
    #                 ax.imshow(img_pred)
    #                 ax.scatter(kp_pred[vis, 0], kp_pred[vis, 1], c="red")
    #                 ax.scatter(kps[vis, 0], kps[vis, 1], c="g")
    #                 fig.savefig(img_file)
    #                 plt.close()


def main(opts):
    torch.manual_seed(0)
    np.random.seed(0)
    print("Training initialization...")
    trainer = ShapeTrainer(opts)
    trainer.init_training()
    print("Done initializing training.")
    trainer.train()
    # trainer.eval()


if __name__ == "__main__":
    fname_data = os.path.join(CONFIG_DIR, CONFIG_DATA)
    with open(fname_data) as f:
        opts_data = json.load(f)
    fname_model = os.path.join(CONFIG_DIR, CONFIG_MODEL)
    with open(fname_model) as f:
        opts_model = json.load(f)
    opts = opts_data | opts_model
    main(opts)
