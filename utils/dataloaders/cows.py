import os
import json
import torch
import imageio
import numpy as np
import pickle as pkl
import torchvision.transforms
from glob import glob
from tqdm import tqdm
from data import smal_base as base_data
from utils import image as img_util


class AnimalKey(base_data.BaseDataset):
    def __init__(self, opts, filter_key=None, filter_name=None, is_train=False):
        super(AnimalKey, self).__init__(opts, filter_key=filter_key)
        self.opts = opts
        self.filter_key = filter_key
        self.filter_name = filter_name

        if filter_name is None:
            image_paths = glob(
                os.path.join(self.opts["data_dir"], self.opts["image_file_string"])
            )
        else:
            image_paths = glob(
                os.path.join(
                    self.opts["data_dir"], filter_name + self.opts["image_file_string"]
                )
            )
        if is_train:
            image_paths = [
                image_path
                for image_path in image_paths
                if os.path.basename(image_path).split("_")[0]
                not in self.opts["test_list"]
            ]
        else:
            image_paths = [
                image_path
                for image_path in image_paths
                if os.path.basename(image_path).split("_")[0] in self.opts["test_list"]
            ]

        bboxes = json.load(open(opts["bboxes_path"]))
        num_images = np.min([len(image_paths), self.opts["num_images"]])
        images = image_paths[:num_images]
        self.anno = [None] * num_images
        self.anno_camera = [None] * len(images)

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        for (i, img_path) in tqdm(enumerate(images)):
            mask_name = os.path.basename(img_path).replace(".jpg", ".png")
            mask_path = os.path.join(os.path.dirname(img_path), "../mask", mask_name)
            mask = imageio.imread(mask_path) / 255.0
            mask = mask[:, :, 0]
            vert_idxs = np.argwhere(mask.sum(axis=0) != 0)
            hor_idxs = np.argwhere(mask.sum(axis=1) != 0)
            bbox = [
                np.min(vert_idxs),
                np.min(hor_idxs),
                np.max(vert_idxs) - np.min(vert_idxs),
                np.max(hor_idxs) - np.min(hor_idxs),
            ]
            bboxes[os.path.split(img_path)[-1].split(".")[0]] = bbox

            kp, vis = img_util.load_keypoints(img_path, self.opts)

            img, kp, mask = img_util.get_image_with_annotations(
                img_path,
                kp,
                bboxes,
                self.opts,
                img_cropped=False,
                mask_cropped=False,
            )

            self.anno[i] = {
                "img_path": img_path,
                "keypoints": np.hstack([kp, vis]),
                "bbox": bboxes[os.path.split(img_path)[-1].split(".")[0]],
            }

            if self.opts["preload_image"]:
                img = torch.from_numpy(img)
                self.anno[i]["img_orig"] = img.clone()
                self.anno[i]["img"] = self.resnet_transform(img)

            if self.opts["preload_mask"]:
                self.anno[i]["mask"] = mask

        self.num_imgs = len(self.anno)
        print("%d images" % self.num_imgs)


def data_loader(opts, shuffle=True, filter_name=None):
    return base_data.base_loader(
        AnimalKey,
        opts["batch_size"],
        opts,
        filter_key=None,
        shuffle=shuffle,
        filter_name=filter_name,
    )