import os
import json
import scipy
import imageio
import torch
import numpy as np
import torchvision.transforms
from glob import glob
from tqdm import tqdm

# from data import smal_base as base_data
from data import base_data_old as base_data


# -------------- Dataset ------------- #
# ------------------------------------ #
class TigDog(base_data.BaseDataset):
    """
    TigDog Data loader
    """

    def __init__(self, opts, train=True, filter_key=None, filter_name=None):
        super(TigDog, self).__init__(opts, filter_key=filter_key)
        self.opts = opts
        animal = opts["animal"]

        if filter_name is None:
            images = glob(
                os.path.join(opts["tigdog_dir"], animal, opts["image_file_string"])
            )
        else:
            images = glob(
                os.path.join(
                    opts["tigdog_dir"],
                    animal,
                    filter_name + opts["image_file_string"],
                )
            )  # TODO: check

        td_ranges = scipy.io.loadmat(
            os.path.join(self.opts["tigdog_dir"], "ranges", animal, "ranges.mat")
        )["ranges"]

        # create mapping between frame names and annotation names
        f2anno = {}
        np.apply_along_axis(
            lambda row: f2anno.update({val: row for val in range(row[1], row[2] + 1)}),
            1,
            td_ranges,
        )

        self.anno = []

        if train:
            images = images[: int(len(images) * 0.9)]
        else:
            images = images[int(len(images) * 0.9) :]

        num_images = 0
        for (i, img_path) in tqdm(enumerate(images)):
            if num_images == opts["num_images"]:
                break

            img_base = int(os.path.basename(img_path).split(".")[0])

            anno_range = f2anno[img_base]
            anno_path = os.path.join(
                self.opts["tigdog_dir"],
                "landmarks",
                animal,
                str(anno_range[0]) + ".mat",
            )
            if not os.path.exists(anno_path):
                continue

            anno_idx = img_base - anno_range[1]
            kp, vis = scipy.io.loadmat(anno_path)["landmarks"][anno_idx][0][0][0]
            kp_vis = np.hstack([kp, vis]).astype(float)

            mask_path = os.path.join(
                self.opts["tigdog_dir"],
                f"{animal}Seg",
                f"seg{str(anno_range[0])}.mat",
            )

            mask = scipy.io.loadmat(mask_path)["segments"][anno_idx][0]
            if mask.sum() == 0:
                print("Empty mask: frame", img_base)
                continue

            vert_idxs = np.argwhere(mask.sum(axis=0) != 0)
            hor_idxs = np.argwhere(mask.sum(axis=1) != 0)
            bbox = [
                np.min(vert_idxs),
                np.min(hor_idxs),
                np.max(vert_idxs) - np.min(vert_idxs),
                np.max(hor_idxs) - np.min(hor_idxs),
            ]

            if bbox[2] < opts["img_size"] / 2 or bbox[3] < opts["img_size"] / 2:
                continue  # omit very small images

            # img = imageio.imread(img_path) / 255.0
            # img = np.transpose(img, (2, 0, 1))
            # img = torch.from_numpy(img)[
            #     :, bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]
            # ]
            self.anno.append(
                {
                    "img_path": img_path,
                    #"img_orig": img.clone(),
                    #"img": self.resnet_transform(img),
                    "mask_path": mask_path,
                    "keypoints": kp_vis,
                    "bbox": bbox,
                    "anno_idx": anno_idx,
                }
            )
            num_images += 1

        self.anno_camera = [None] * num_images
        self.num_imgs = len(self.anno)
        print("%d images" % self.num_imgs)


def data_loader(opts, shuffle=True):
    return base_data.base_loader(
        TigDog,
        opts,
        shuffle=shuffle,
    )


if __name__ == "__main__":
    with open("configs/tigdog_data.json") as f:
        opts = json.load(f)
    dataset = TigDog(opts)