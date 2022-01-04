"""
Evaluation on the test set.
"""

import os

import json
import shutil
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Dict, List, Any, Optional

from src import loss_utils
from data.dogs import StanfordExtra
from data.zebra import ZebraDataset
from data.cows import AnimalKey
from data.tigdog import TigDog
from src.mesh_predictor import MeshPredictor
from utils import image as img_util
from utils.smal_vis import VisRenderer


# TODO: pass as flags
CONFIG_DIR = "configs/"
CONFIG_EVAL = "evaluation_params_zebra.json"
CONFIG_DATA = "zebra_data.json"
KP_COLOR = (255, 0, 0)
ALPHA = (0.01, 0.02, 0.05, 0.1, 0.15)


class ShapeEvaluator:
    def __init__(self, opts):
        self.opts = opts
        # TODO: make utils function for this
        if os.path.exists(self.opts["evaluation_dir"]):
            shutil.rmtree(self.opts["evaluation_dir"])
        os.makedirs(self.opts["evaluation_dir"])
        if os.path.exists(self.opts["visualizations_dir"]):
            shutil.rmtree(self.opts["visualizations_dir"])
        os.makedirs(self.opts["visualizations_dir"])

        self.predictor = MeshPredictor(opts, is_train=False)
        self.predictor.to(self.opts["gpu_id"])
        self.predictor.eval()

        if opts["dataset"] == "stanford_extra":
            self.dataset = StanfordExtra(opts, use_augmentation=False, is_train=False)
        elif opts["dataset"] == "grevy_zebra":
            self.dataset = ZebraDataset(opts)
        elif opts["dataset"] == "tigdog":
            self.dataset = TigDog(opts, train=False)
        elif opts["dataset"] == "animalkey":
            self.dataset = AnimalKey(opts, is_train=False)
        else:
            raise "Unknown Dataset"
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

    def visualize(
        self,
        batch,
        preds,
        out_name: str,
    ):
        img = batch["img_orig"].cpu().numpy().squeeze()
        img = np.transpose(img, (1, 2, 0))
        if "mask" in batch:
            mask_gt = batch["mask"].cpu().numpy().squeeze()
        else:
            mask_gt = None
        kp_gt = batch["keypoints"].cpu().numpy()[0, :, :2]
        vis = batch["keypoints"][0, :, 2]

        kp_pred = preds["kp_pred"].cpu().numpy().squeeze()
        mask_pred = preds["mask_pred"].cpu().numpy().squeeze()
        vert = preds["verts_pred"][0]
        cam = preds["camera_pred"][0]
        shape_pred = self.predictor.vis_renderer(vert, opts["gpu_id"], cams=cam)
        if self.opts["pipeline_version"] == "smalst":
            shape_pred = np.flipud(shape_pred)
            mask_pred = np.flipud(mask_pred)

        Iov = (0.5 * 255 * img + 0.5 * shape_pred.astype(np.uint8)).astype(np.uint8)
        imageio.imwrite(
            os.path.join(self.opts["evaluation_dir"], "shape_ov_" + out_name), Iov
        )
        imageio.imwrite(
            os.path.join(self.opts["evaluation_dir"], "shape_" + out_name), shape_pred
        )

        nrows = 4 if mask_gt is not None else 3
        _, ax = plt.subplots(1, nrows)
        ax[0].imshow(img)
        ax[0].set_title("input")
        ax[0].axis("off")
        ax[1].imshow(img)
        ax[1].imshow(shape_pred, alpha=0.7)
        if vis.sum() > 0:
            idx = np.where(vis)
            ax[1].scatter(kp_gt[idx, 0], kp_gt[idx, 1], c="k", s=2)
            ax[1].scatter(kp_pred[idx, 0], kp_pred[idx, 1], c="r", s=2)
        ax[1].set_title("pred mesh")
        ax[1].axis("off")
        ax[2].imshow(mask_pred, cmap="gray")
        ax[2].axis("off")
        ax[2].set_title("pred mask")
        if nrows == 4:
            ax[3].imshow(mask_gt, cmap="gray")
            ax[3].axis("off")
            ax[3].set_title("gt mask")
        # out_name = os.path.basename(batch["img_path"][0])
        out_path = os.path.join(opts["visualizations_dir"], out_name)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

    @staticmethod
    def compute_pck(kp_pred, kp_vis, mask=None, img_size=None, alpha=0.10):
        # Assuming batch_size = 1 here (evaluation only)
        kp_pred = kp_pred.cpu().numpy()[0, ...]
        kp, vis = kp_vis[0, :, :2].cpu().numpy(), kp_vis[0, :, 2].cpu().numpy()
        if mask is not None:
            kp_diffs = np.linalg.norm(kp_pred - kp, axis=-1)[np.where(vis == 1)[0]]
            kp_diffs /= np.sqrt(mask.sum().item())
        else:
            kp_diffs = np.linalg.norm(kp_pred / img_size - kp / img_size, axis=-1)[
                np.where(vis == 1)[0]
            ]
        if vis.sum() > 0:
            pck = np.mean(kp_diffs < alpha)
        else:
            pck = 0.0
        return pck

    @staticmethod
    def compute_iou(batch, preds):
        mask_gt = batch["mask"].cpu().numpy().squeeze()
        mask_pred = preds["mask_pred"].cpu().numpy().squeeze()
        # border_mask = batch["img_border_mask"].cpu().numpy().squeeze()
        # mask_gt = mask_gt * border_mask + mask_pred * (1 - border_mask)
        intersection = np.sum(mask_gt * mask_pred)
        union = np.sum(mask_gt) + np.sum(mask_pred) - np.sum(mask_gt * mask_pred)
        return intersection / union

    def evaluate_all_images(
        self, update_freq=16
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dataset_name = self.opts["dataset"]
        pck_total = np.zeros((len(self.dataset)))
        # pck_by_part = {
        #     group: np.zeros((len(self.dataset))) for group in self.opts["kp_groups"]
        # }
        iou_total = np.zeros(len(self.dataset))
        tqdm_iterator = tqdm(self.data_loader, desc="Eval", total=len(self.data_loader))
        for (step, batch) in enumerate(tqdm_iterator):
            with torch.no_grad():
                preds = self.predictor.forward(batch["img"])
                # if dataset_name in ("stanford_extra", "grevy_zebra"):
                # transform data from [-1, +1] range
                batch["keypoints"][0, :, :2] = (
                    (batch["keypoints"][0, :, :2] + 1.0) * 0.5
                ) * self.opts["img_size"]

                if self.opts["use_keypoints"]:
                    pck = self.compute_pck(
                        kp_pred=preds["kp_pred"],
                        kp_vis=batch["keypoints"],
                        img_size=self.opts["img_size"],
                        # mask=batch["mask"],
                    )
                    pck_total[step] = pck

                if self.opts["use_masks"]:
                    iou = self.compute_iou(batch, preds)
                    iou_total[step] = iou

                self.visualize(batch, preds, out_name=f"visualization_{step}.png")
                # break
                # if (step + 1) % update_freq == 0:
                #     pck_cur = pck_total[step - update_freq + 1 : step + 1].mean()
                #     iou_cur = iou_total[step - update_freq + 1 : step + 1].mean()
                #     tqdm_iterator.desc = (
                #         f"PCK: {round(pck_cur, 2)}, IoU: {round(iou_cur, 2)}"
                #     )
                #     tqdm_iterator.update()

        print("\nTotal Results for {dataset_name}:")
        print("PCK@0.1:", pck_total.mean())
        print("IoU", iou_total.mean())


def main(opts):
    # tex_mask = imageio.imread(opts["texture_mask_path"]) / 255.0
    # bboxes = json.load(open(opts["bboxes_path"]))

    # image_paths = sorted(glob(opts["img_path"] + "*" + opts["img_ext"]))
    # n_images = 2 * len(image_paths) if opts["mirror"] else len(image_paths)
    # print("Test set size:", n_images)

    evaluator = ShapeEvaluator(opts)
    evaluator.evaluate_all_images()

    # if opts["use_annotations"]:
    #     print("Total PCK")
    #     print(f"{np.mean(err_tot, axis=0)} +- {np.std(err_tot, axis=0)}")
    #     print("Total Overlap")
    #     print(f"{np.mean(overlap, axis=0)} +- {np.std(overlap, axis=0)}")
    #     print("Total IOU")
    #     print(f"{np.mean(iou, axis=0)} +- {np.std(iou, axis=0)}")


if __name__ == "__main__":
    fname_data = os.path.join(CONFIG_DIR, CONFIG_DATA)
    with open(fname_data) as f:
        opts_data = json.load(f)
    fname_eval = os.path.join(CONFIG_DIR, CONFIG_EVAL)
    with open(fname_eval) as f:
        opts_eval = json.load(f)
    opts = opts_data | opts_eval
    main(opts)
