"""
Original author: Silvia Zuffi (https://github.com/silviazuffi/smalst)
Refactoring and Python 3 porting: Valeria Iegorova (https://github.com/iegorval/smalmv)
MIT licence
"""
import numpy as np
import imageio
import scipy.io as sio
import torchvision
import torch
from src import smal_predictor as pred_util
from utils import image as img_util
from glob import glob
from torch.autograd import Variable


def preprocess_image(img_path, img_size=256, kp=None, border=20):
    img = imageio.imread(img_path) / 255.0
    img = img[:, :, :3]

    # Scale the max image size to be img_size
    scale_factor = float(img_size - 2 * border) / np.max(img.shape[:2])
    img, _ = img_util.resize_img(img, scale_factor)

    # Crop img_size x img_size from the center
    img, center = img_util.crop_image(img)
    img, _, bbox = img_util.resize_img(img, 256 / 257.0)

    # Transpose the image to 3xHxW
    img = np.transpose(img, (2, 0, 1))

    if kp is not None:
        kp = kp * scale_factor
        kp[:, 0] -= bbox[0]
        kp[:, 1] -= bbox[1]

    return img, kp


def smal_mesh_eval(num_train_epoch, opts):
    opts["num_train_epoch"] = num_train_epoch
    img_path = opts["img_path_val"]
    images = sorted(glob(img_path + "*.jpg"))
    anno_path = opts["anno_path_val"]
    annotations = sorted(glob(anno_path + "*_ferrari-tail-face.mat"))
    n_images = len(images)

    batch_size = opts["batch_size"]
    opts["batch_size"] = 1

    predictor = pred_util.MeshPredictor(opts)
    tot_pose_err = 0

    err_tot = np.zeros(n_images)
    for idx, img_path in enumerate(images):
        res = sio.loadmat(annotations[idx], squeeze_me=True, struct_as_record=False)
        res = res["annotation"]
        kp = res.kp.astype(float)
        invisible = res.invisible
        vis = np.atleast_2d(~invisible.astype(bool)).T
        # landmarks = np.hstack((kp, vis))
        # names = [str(res.names[i]) for i in range(len(res.names))]

        img, kp = preprocess_image(img_path, img_size=opts["img_size"], kp=kp)
        batch = {"img": torch.Tensor(np.expand_dims(img, 0))}
        outputs = predictor.predict(batch)

        kp_pred = (
            (outputs["kp_pred"].cpu().detach().numpy()[0, :, :] + 1.0) * 128
        ).astype(int)
        kp_err = np.sum(np.abs(kp - kp_pred) * vis) / np.sum(vis)
        tot_pose_err += kp_err
        err_tot[idx] = kp_err

    opts["batch_size"] = batch_size
    return tot_pose_err / n_images


def set_input(self, batch):
    resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    opts = self.opts

    img_tensor = batch["img"].clone().type(torch.FloatTensor)

    input_img_tensor = batch["img"].type(torch.FloatTensor)
    for b in range(input_img_tensor.size(0)):
        input_img_tensor[b] = resnet_transform(input_img_tensor[b])

    self.input_imgs = Variable(
        input_img_tensor.cuda(device=opts["gpu_id"]), requires_grad=False
    )
    self.imgs = Variable(img_tensor.cuda(device=opts["gpu_id"]), requires_grad=False)


def collect_outputs(self):
    outputs = {
        "pose_pred": self.pose.data,
        "kp_pred": self.kp_pred.data,
        "verts": self.pred_v.data,
        "kp_verts": self.kp_verts.data,
        "cam_pred": self.cam_pred.data,
        "mask_pred": self.mask_pred.data,
        "faces": self.faces,
        "delta_v_pred": self.delta_v.data,
        "trans_pred": self.trans.data,
        "kp_2D_pred": self.kp_2D_pred,
        "f": self.faces,
        "v": self.smal_verts,
    }
    if self.opts["use_smal_betas"]:
        outputs["betas_pred"] = self.betas.data
    if self.opts["texture"]:
        outputs["texture"] = self.textures
        outputs["texture_pred"] = self.texture_pred.data
        outputs["uv_image"] = self.uv_images.data
        outputs["uv_flow"] = self.uv_flows.data
    if self.opts["predict_ambient_occlusion"]:
        outputs["occ_pred"] = self.occ_map

    return outputs
