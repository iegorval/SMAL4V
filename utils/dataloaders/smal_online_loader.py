import os
import json
import torch
import numpy as np
from glob import glob
from tqdm import tqdm

# from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import image as img_util


# ------------ Dataset ----------- #
# ------------------------------------ #
class VideoSequence(Dataset):
    """
    Temporal data loading class.


    Attributes
    ----------

    Methods
    -------
    """

    def __init__(self, opts, seq_name, cnn_encoder=None, train=False):
        self.opts = opts
        self.train = train
        self.seq_path = os.path.join(self.opts["sequences_dir"], seq_name)
        self.cnn_encoder = cnn_encoder
        self.cnn_encoder.to(device=self.opts["gpu_id"])
        self.cnn_encoder.eval()
        self.load_sequence()

    def load_sequence(self):
        kps_dir = os.path.join(self.seq_path, self.opts["keypoints_dir"])
        masks_dir = os.path.join(self.seq_path, self.opts["masks_dir"])
        bboxes = json.load(open(os.path.join(self.seq_path, self.opts["bboxes_path"])))
        img_paths = glob(os.path.join(self.seq_path, "*" + self.opts["img_ext"]))
        img_paths.sort(key=lambda s: int(s.split(".")[0].split("_")[-1]))
        images, kps, masks, enc_feats = [], [], [], []

        self.seq_idxs = []
        for (i, img_path) in enumerate(tqdm(img_paths)):
            img_name = os.path.split(img_path)[-1]
            if img_name not in bboxes:
                raise "Bounding box missing (remove the image or add bounding box)"

            full_window = (i + self.opts["seq_length"]) < len(img_paths)
            if i % self.opts["seq_stride"] == 0 and full_window:
                self.seq_idxs.append((i, i + self.opts["seq_length"]))
                # print((i, i + self.opts["seq_length"], len(img_paths)))

            kp, vis = img_util.load_keypoints(img_path, self.opts, kps_dir)
            img, kp, mask = img_util.get_image_with_annotations(
                img_path,
                kp,
                bboxes,
                self.opts,
                masks_dir,
                img_cropped=False,
                mask_cropped=False,
            )

            if kp is None:
                kp_vis = np.zeros((self.opts["num_kps"], 3))
            else:
                kp_vis = np.hstack([kp, vis])
            kps.append(kp_vis)

            if mask is None:
                mask = np.zeros((self.opts["img_size"], self.opts["img_size"]))
            masks.append(mask)

            # Precompute features from CNN encoder
            images.append(img)
            if self.cnn_encoder is not None:
                img = (
                    torch.Tensor(img[np.newaxis, ...])
                    .type(torch.FloatTensor)
                    .to(self.opts["gpu_id"])
                )
                with torch.no_grad():
                    _, enc_feat, _ = self.cnn_encoder.forward(img, fg_img=None)
                # images.append(feat.cpu().numpy())
                enc_feats.append(enc_feat.cpu().numpy())
            else:
                images.append(img)

            # i += 1
            # if i == 30:
            #     break

        self.seq = {
            "images": np.stack(images),
            "keypoints": np.stack(kps),
            "masks": np.stack(masks),
        }
        # len(seq_idxs) - self.opts["seq_length"]
        # self.seq_idxs = seq_idxs[: len(seq_idxs) - self.opts["seq_length"]]
        # for i in range(len(seq_idxs) - self.opts["seq_length"], len(seq_idxs)):
        #     if seq_idxs[i][1] + self.opts["seq_length"] >= len(self.seq["images"]):
        #         self.seq_idxs.append(seq_idxs[i])

        if enc_feats:
            self.seq["enc_features"] = np.stack(enc_feats)

    def __len__(self):
        return len(self.seq_idxs)
        # return len(self.seq["images"]) // (
        #     self.opts["seq_length"] * self.opts["batch_size"]
        # )

    def __getitem__(self, index):
        # seq_indices = (
        #     index * self.opts["seq_length"],
        #     (index + 1) * self.opts["seq_length"],
        # )
        seq_indices = self.seq_idxs[index]
        window = {
            "images": self.seq["images"][seq_indices[0] : seq_indices[1]],
            "keypoints": self.seq["keypoints"][seq_indices[0] : seq_indices[1]],
            "masks": self.seq["masks"][seq_indices[0] : seq_indices[1]],
        }
        if "enc_features" in self.seq:
            window["enc_features"] = self.seq["enc_features"][
                seq_indices[0] : seq_indices[1]
            ]
        return window


# ------------ Data Loader ----------- #
# ------------------------------------ #
def data_loader(opts, seq_name, cnn_encoder=None, shuffle=True):
    dset = VideoSequence(opts, seq_name, cnn_encoder)
    return DataLoader(
        dset,
        shuffle=shuffle,
        batch_size=opts["batch_size"],
        drop_last=True,
    )