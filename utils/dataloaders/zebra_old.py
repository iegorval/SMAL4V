"""

"""
import os
import json
import imageio
import torch
import scipy
import numpy as np
import pickle as pkl
from glob import glob
from torch.utils.data import Dataset
from data import smal_base as base_data
from utils import transformations

# -------------- Dataset ------------- #
# ------------------------------------ #
class ZebraDataset(base_data.BaseDataset):
    """
    Zebra Data loader
    """

    def __init__(self, opts, filter_key=None, filter_name=None):
        super(ZebraDataset, self).__init__(opts, filter_key=filter_key)
        self.opts = opts
        self.data_cache_dir = self.opts["zebra_cache_dir"]
        self.filter_key = filter_key
        self.filter_name = filter_name

        self.data_dir = self.opts["data_dir"]
        self.img_dir = os.path.join(self.data_dir, "images")

        if filter_name is None:
            images = glob(os.path.join(self.img_dir, self.opts["image_file_string"]))
        else:
            images = glob(
                os.path.join(self.img_dir, filter_name + self.opts["image_file_string"])
            )
        num_images = np.min([len(images), self.opts["num_images"]])

        images = images[:num_images]
        self.anno = [None] * num_images
        self.anno_camera = [None] * len(images)
        for (i, img) in enumerate(images):
            anno_path = os.path.join(
                self.data_dir,
                "annotations/%s.pkl" % os.path.splitext(os.path.basename(img))[0],
            )
            if os.path.exists(anno_path):
                self.anno[i] = pkl.load(open(anno_path, "rb"), encoding="latin1")
                self.anno[i]["mask_path"] = os.path.join(
                    self.data_dir,
                    "bgsub/%s.png" % os.path.splitext(os.path.basename(img))[0],
                )
                self.anno[i]["img_path"] = img
                uv_flow_path = os.path.join(
                    self.data_dir,
                    "uvflow/%s.pkl" % os.path.splitext(os.path.basename(img))[0],
                )
                if os.path.exists(uv_flow_path):
                    self.anno[i]["uv_flow_path"] = uv_flow_path

                # In case we have the texture map
                if "texture_map_filename" in self.anno[i].keys():
                    if self.opts["use_per_file_texmap"]:
                        self.anno[i]["texture_map"] = os.path.join(
                            self.data_dir,
                            "texmap/%s.png"
                            % os.path.splitext(os.path.basename(img))[0],
                        )
                    else:
                        self.anno[i]["texture_map"] = os.path.join(
                            self.data_dir,
                            "texture_maps/%s" % self.anno[i]["texture_map_filename"],
                        )

                # Add a column to the keypoints in case the visibility is not defined
                if self.anno[i]["keypoints"].shape[1] < 3:
                    self.anno[i]["keypoints"] = np.column_stack(
                        [
                            self.anno[i]["keypoints"],
                            np.ones(self.anno[i]["keypoints"].shape[0]),
                        ]
                    )

                self.anno_camera[i] = {
                    "flength": self.anno[i]["flength"],
                    "trans": np.zeros(2, dtype=float),
                }
                self.kp_perm = np.array(range(self.anno[0]["keypoints"].shape[0]))

                if self.opts["preload_image"]:
                    self.anno[i]["img"] = (
                        scipy.misc.imread(self.anno[i]["img_path"]) / 255.0
                    )
                if self.opts["preload_texture_map"]:
                    texture_map = scipy.misc.imread(self.anno[i]["texture_map"]) / 255.0
                    self.anno[i]["texture_map_data"] = np.transpose(
                        texture_map, (2, 0, 1)
                    )
                if self.opts["preload_mask"]:
                    self.anno[i]["mask"] = (
                        scipy.misc.imread(self.anno[i]["mask_path"]) / 255.0
                    )
                if self.opts["preload_uvflow"]:
                    uvdata = pkl.load(open(self.anno[i]["uv_flow_path"], "rb"))
                    uv_flow = uvdata["uv_flow"].astype(np.float32)
                    uv_flow[:, :, 0] = uv_flow[:, :, 0] / (uvdata["img_h"] / 2.0)
                    uv_flow[:, :, 1] = uv_flow[:, :, 1] / (uvdata["img_w"] / 2.0)
                    self.anno[i]["uv_flow"] = uv_flow

            else:
                mask_path = os.path.join(
                    self.data_dir,
                    "bgsub/%s.png" % os.path.splitext(os.path.basename(img))[0],
                )
                if os.path.exists(mask_path):
                    self.anno[i] = {
                        "mask_path": mask_path,
                        "img_path": img,
                        "keypoints": None,
                        "uv_flow": None,
                    }
                else:
                    self.anno[i] = {
                        "mask_path": None,
                        "img_path": img,
                        "keypoints": None,
                        "uv_flow": None,
                    }
                self.anno_camera[i] = {"flength": None, "trans": None}
        self.num_imgs = len(self.anno)

        print("%d images" % self.num_imgs)


# ----------- Data Loader ----------#
# ----------------------------------#
def data_loader(opts, shuffle=True, filter_name=None):
    return base_data.base_loader(
        ZebraDataset,
        opts["batch_size"],
        opts,
        filter_key=None,
        shuffle=shuffle,
        filter_name=filter_name,
    )


def kp_data_loader(batch_size, opts):
    return base_data.base_loader(ZebraDataset, batch_size, opts, filter_key="kp")


def mask_data_loader(batch_size, opts):
    return base_data.base_loader(ZebraDataset, batch_size, opts, filter_key="mask")


def texture_map_data_loader(batch_size, opts):
    return base_data.base_loader(
        ZebraDataset, batch_size, opts, filter_key="texture_map"
    )
