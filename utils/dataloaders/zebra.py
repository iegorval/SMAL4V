import os
import json
import torch
import numpy as np
import pickle as pkl
import torchvision.transforms
from glob import glob
from tqdm import tqdm
from data import smal_base as base_data
from utils import image as img_util


class ZebraDataset(base_data.BaseDataset):
    def __init__(self, opts, filter_key=None, filter_name=None):
        super(ZebraDataset, self).__init__(opts, filter_key=filter_key)
        self.opts = opts
        self.filter_key = filter_key
        self.filter_name = filter_name

        if filter_name is None:
            img_paths = glob(
                os.path.join(self.opts["data_dir"], self.opts["image_file_string"])
            )
        else:
            img_paths = glob(
                os.path.join(
                    self.opts["data_dir"], filter_name + self.opts["image_file_string"]
                )
            )
        bboxes = json.load(open(opts["bboxes_path"]))
        num_images = np.min([len(img_paths), self.opts["num_images"]])
        img_paths = img_paths[:num_images]
        self.anno = [None] * num_images * 2
        self.anno_camera = [None] * len(img_paths) * 2

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        for (i, img_path) in tqdm(enumerate(img_paths)):
            kp, vis = img_util.load_keypoints(img_path, opts)
            img, kp, mask = img_util.get_image_with_annotations(
                img_path, kp, bboxes, opts
            )
            bbox = bboxes[os.path.split(img_path)[-1].split(".")[0]]
            self.anno[2 * i] = {
                "img_path": img_path,
                "keypoints": np.hstack([kp, vis]),
                "bbox": bbox,
            }
            kp_m, vis_m = self.mirror_keypoints(
                kp, vis, self.opts["img_size"], self.opts["dx2sx"]
            )
            self.anno[2 * i + 1] = {
                "img_path": img_path,
                "keypoints": np.hstack([kp_m, vis_m]),
                "bbox": None,
            }

            if self.opts["preload_image"]:
                img = torch.from_numpy(img)
                img_m = torch.flip(img, dims=[2])
                self.anno[2 * i]["img_orig"] = img.clone()
                self.anno[2 * i]["img"] = self.resnet_transform(img)
                self.anno[2 * i + 1]["img_orig"] = img_m.clone()
                self.anno[2 * i + 1]["img"] = self.resnet_transform(img_m)

            if self.opts["preload_mask"]:
                self.anno[2 * i]["mask"] = mask
                self.anno[2 * i + 1]["mask"] = mask[:, ::-1]

        self.num_imgs = len(self.anno)
        print("%d images" % self.num_imgs)

    @staticmethod
    def mirror_keypoints(kp: np.ndarray, vis: np.ndarray, img_width: int, dx2sx):
        kp_m = np.zeros_like(kp)
        kp_m[:, 0] = img_width - kp[dx2sx, 0] - 1
        kp_m[:, 1] = kp[dx2sx, 1]
        vis_m = np.zeros_like(vis)
        vis_m[:] = vis[dx2sx]
        return kp_m, vis_m


def data_loader(opts, shuffle=True, filter_name=None):
    return base_data.base_loader(
        ZebraDataset,
        opts["batch_size"],
        opts,
        filter_key=None,
        shuffle=shuffle,
        filter_name=filter_name,
    )