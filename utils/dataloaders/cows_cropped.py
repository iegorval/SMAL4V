import os
import json
import scipy
import numpy as np
import pickle as pkl
from glob import glob
from data import smal_base as base_data


class CowsDataset(base_data.BaseDataset):
    def __init__(self, opts, filter_key=None, filter_name=None):
        super(CowsDataset, self).__init__(opts, filter_key=filter_key)
        self.opts = opts
        self.data_cache_dir = self.opts["cows_cache_dir"]
        self.filter_key = filter_key
        self.filter_name = filter_name

        if filter_name is None:
            images = glob(
                os.path.join(self.opts["cows_dir"], self.opts["image_file_string"])
            )
        else:
            images = glob(
                os.path.join(
                    self.opts["cows_dir"], filter_name + self.opts["image_file_string"]
                )
            )
        bboxes = json.load(open(opts["bboxes_path"]))
        num_images = np.min([len(images), self.opts["num_images"]])
        images = images[:num_images]
        self.anno = [None] * num_images
        self.anno_camera = [None] * len(images)
        for (i, img) in enumerate(images):
            anno_path = os.path.join(
                self.opts["cows_dir"],
                "annotations/%s.pkl" % os.path.splitext(os.path.basename(img))[0],
            )
            if os.path.exists(anno_path):
                kp = pkl.load(open(anno_path, "rb")).astype(np.float64)
                self.anno[i] = {
                    "img_path": img,
                    "keypoints": kp,
                    "bbox": bboxes[img.split("/")[-1]],  # TODO: path.split
                }
                mask_path = os.path.join(
                    self.opts["cows_dir"],
                    "masks/%s.png" % os.path.splitext(os.path.basename(img))[0],
                )
                if os.path.exists(mask_path):
                    self.anno[i]["mask_path"] = mask_path
                else:
                    self.anno[i]["mask_path"] = None
                self.kp_perm = np.array(range(self.anno[0]["keypoints"].shape[0]))

                if self.opts["preload_image"]:
                    self.anno[i]["img"] = (
                        scipy.misc.imread(self.anno[i]["img_path"]) / 255.0
                    )

                if self.opts["preload_mask"]:
                    self.anno[i]["mask"] = (
                        scipy.misc.imread(self.anno[i]["mask_path"]) / 255.0
                    )

        self.num_imgs = len(self.anno)
        print("%d images" % self.num_imgs)


def data_loader(opts, shuffle=True, filter_name=None):
    return base_data.base_loader(
        CowsDataset,
        opts["batch_size"],
        opts,
        filter_key=None,
        shuffle=shuffle,
        filter_name=filter_name,
    )