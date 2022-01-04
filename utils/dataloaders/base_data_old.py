import os
import torch
from re import I
import scipy.io
import imageio
import numpy as np
import pickle as pkl
import torchvision.transforms
from glob import glob
from copy import deepcopy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import image as image_utils


# ------------ Dataset ----------- #
# ------------------------------------ #
class BaseDataset(Dataset):
    """
    Base data loading class.

    Attributes
    ----------

    Methods
    -------

    Should output:
        - img: B X 3 X H X W
        - kp: B X nKp X 2
        - mask: B X H X W
        # Silvia - sfm_pose: B X 7 (s, tr, q)
        - camera_params: B X 4 (s, tr)
        (kp, sfm_pose) correspond to image coordinates in [-1, 1]
    """

    def __init__(self, opts, filter_key=None):
        # Child class should define/load:
        # self.kp_perm
        # self.img_dir
        # self.anno
        # self.anno_camera
        self.opts = opts
        self.filter_key = filter_key

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if "use_smal_betas" in opts and not opts["use_smal_betas"]:
            # We need to load the blendshapes
            model_path = os.path.join(self.opts["model_dir"], self.opts["model_name"])
            with open(model_path, "rb") as f:
                dd = pkl.load(f, encoding="latin1")
                num_betas = dd["shapedirs"].shape[-1]
                self.shapedirs = np.reshape(dd["shapedirs"], [-1, num_betas]).T

    def forward_img(self, index):
        data = deepcopy(self.anno[index])
        if self.anno_camera[index] is not None:
            data_sfm = self.anno_camera[index].copy()
            camera_params = [np.copy(data_sfm["flength"]), np.zeros(2)]
        else:
            camera_params = [
                np.asarray([2700.0]),
                np.zeros(2),
            ]  # TODO: add param for FDEFAULT

        img_path = data["img_path"]
        if "img" in data.keys():
            img = data["img"]
        else:
            img = np.asarray(imageio.imread(img_path)) / 255.0

        if "texture_map" in data.keys():
            texture_map_path = data["texture_map"]
            if "texture_map_data" in data.keys():
                texture_map = data["texture_map_data"]
            else:
                texture_map = np.asarray(imageio.imread(texture_map_path)) / 255.0
                texture_map = np.transpose(texture_map, (2, 0, 1))
        else:
            texture_map = None

        if "mask_path" in data:
            mask_path = data["mask_path"]
            if "mask" in data.keys():
                mask = data["mask"]
            else:
                if "anno_idx" in data.keys():  # for TigDog
                    mask = scipy.io.loadmat(mask_path)["segments"][data["anno_idx"]][0]
                else:
                    mask = np.asarray(imageio.imread(mask_path)) / 255.0
        elif not self.opts["preload_mask"]:
            mask = None

        if "uv_flow_path" in data.keys():
            uv_flow_path = data["uv_flow_path"]
            if "uv_flow" in data.keys():
                uv_flow = data["uv_flow"]
            else:
                uvdata = pkl.load(open(uv_flow_path, "rb"), encoding="latin1")
                uv_flow = uvdata["uv_flow"].astype(np.float32)
                uv_flow[:, :, 0] = uv_flow[:, :, 0] / (uvdata["img_h"] / 2.0)
                uv_flow[:, :, 1] = uv_flow[:, :, 1] / (uvdata["img_w"] / 2.0)
        else:
            uv_flow = None

        occ_map = None

        kp = data["keypoints"]
        if "trans" in data.keys():
            model_trans = data["trans"].copy()
        else:
            model_trans = None
        if "pose" in data.keys():
            model_pose = data["pose"].copy()
        else:
            model_pose = None
        if "betas" in data.keys():
            model_betas = data["betas"].copy()
        else:
            model_betas = None
        if "delta_v" in data.keys():
            model_delta_v = data["delta_v"].copy()
            if not self.opts["use_smal_betas"]:
                # Modify the deformation to include B*\betas
                nBetas = len(model_betas)
                model_delta_v = model_delta_v + np.reshape(
                    np.matmul(model_betas, self.shapedirs[:nBetas, :]),
                    [model_delta_v.shape[0], model_delta_v.shape[1]],
                )
        else:
            model_delta_v = None

        # Perspective camera needs image center
        camera_params[1][0] = img.shape[1] / 2.0
        camera_params[1][1] = img.shape[0] / 2.0

        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask[:, :, np.newaxis]  # TODO duplicated
            M = mask[:, :, 0]
            xmin = np.min(np.where(M > 0)[1])
            ymin = np.min(np.where(M > 0)[0])
            xmax = np.max(np.where(M > 0)[1])
            ymax = np.max(np.where(M > 0)[0])
        else:
            xmin = 0
            ymin = 0
            xmax = img.shape[1]
            ymax = img.shape[0]

        # Compute bbox
        bbox = np.array([xmin, ymin, xmax, ymax], float)  # TODO: why it is not used?

        if self.opts["border"] > 0:
            assert ("trans" in data.keys()) == False
            assert ("pose" in data.keys()) == False
            assert ("kp" in data.keys()) == False
            # This has to be used only if there are no annotations for the refinement!
            scale_factor = float(
                self.opts["img_size"] - 2 * self.opts["border"]
            ) / np.max(img.shape[:2])
            img, _ = image_utils.resize_img(img, scale_factor)
            if mask is not None:
                mask, _ = image_utils.resize_img(mask, scale_factor)
                # Crop img_size x img_size from the center
            center = np.round(np.array(img.shape[:2]) / 2).astype(int)
            # img center in (x, y)
            center = center[::-1]
            bbox = np.hstack(
                [
                    center - self.opts["img_size"] / 2.0,
                    center + self.opts["img_size"] / 2.0,
                ]
            )

        if kp is not None:
            vis = kp[:, 2] > 0
            kp[vis, :2] -= 1
        else:
            vis = None

        if self.opts["synthetic"]:
            # Peturb bbox
            if self.opts["perturb_bbox"] and mask is not None:
                bbox = image_utils.peturb_bbox(
                    bbox, pf=self.opts["padding_frac"], jf=self.opts["jitter_frac"]
                )
            orig_bbox = bbox[:]
            bbox = image_utils.square_bbox(bbox)
        else:
            # TODO: TO FUNCTION (REPEATED)
            if self.opts["cropped"]:
                full_img_path = os.path.join(
                    os.path.dirname(img_path),
                    "full_size",
                    img_path.split("/")[-1].split(".")[0]
                    + "*."
                    + img_path.split(".")[-1],
                )
            else:
                full_img_path = img_path
            full_img_path = glob(full_img_path)[0]
            if os.path.exists(full_img_path):
                img = imageio.imread(full_img_path) / 255.0
                orig_bbox = [
                    data["bbox"][0],
                    data["bbox"][1],
                    data["bbox"][0] + data["bbox"][2] - 1,
                    data["bbox"][1] + data["bbox"][3] - 1,
                ]
                bbox = image_utils.square_bbox(orig_bbox)
            else:
                raise NotImplementedError  # TODO: black / reflected padding

        if not self.opts["translate_keypoints"]:
            kp[vis, 0] += orig_bbox[0] - bbox[0]
            kp[vis, 1] += orig_bbox[1] - bbox[1]

        # crop image around bbox, translate kps
        # if self.opts['use_bbox'] and mask is not None:
        if not self.opts["is_optimization"]:
            (
                img,
                mask,
                kp,
                camera_params,
                model_trans,
                occ_map,
                uv_flow,
            ) = self.crop_image(
                img, mask, bbox, kp, vis, camera_params, model_trans, occ_map, uv_flow
            )

            # scale image, and mask. And scale kps.
            img, mask, kp, camera_params, occ_map, uv_flow = self.scale_image(
                img, mask, kp, vis, camera_params, occ_map, orig_bbox, uv_flow
            )

        # Normalize kp to be [-1, 1]
        img_h, img_w = img.shape[:2]
        if kp is not None:
            kp_norm = self.normalize_kp(kp, img_h, img_w, self.opts["update_vis"])
        else:
            kp_norm = None

        if not self.opts["use_camera"]:
            focal_length_fix = self.opts["camera_ref"]
            f_scale = focal_length_fix / camera_params[0]
            camera_params[0] *= f_scale
            if model_trans is not None:
                model_trans[2] *= f_scale

        if self.opts["save_training"]:
            img_name = img_path.split("/")[-1].split(".")[0]
            imageio.imwrite(
                os.path.join(self.opts["cropped_dir"], f"{img_name}.crop.png"), img
            )
            if mask is not None:
                imageio.imwrite(
                    os.path.join(self.opts["cropped_dir"], f"{img_name}.crop_mask.png"),
                    mask,
                )
            data = {
                "kp": kp,
                "sfm_pose": camera_params,
                "model_trans": model_trans,
                "model_pose": model_pose,
                "model_betas": model_betas,
            }
            pkl.dump(data, open(img_path + ".crop.pkl", "wb"))
            if uv_flow is not None:
                pkl.dump(uv_flow, open(img_path + "._uv_flow_crop.pkl", "wb"))
            print("saved " + img_path)

        # Finally transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1))
        if mask is not None:
            if len(mask.shape) < 3:  # for compatibility
                mask = mask[:, :, np.newaxis]
            mask = np.transpose(mask, (2, 0, 1))[0, ...][np.newaxis]

        if self.opts["border"] > 0:
            camera_params[1][0] = img.shape[1] / 2.0
            camera_params[1][1] = img.shape[0] / 2.0

        img = torch.from_numpy(img)
        img_orig = img.clone()
        img = self.resnet_transform(img)
        return (
            img,
            img_orig,
            kp,
            kp_norm,
            mask,
            camera_params,
            # texture_map,
            # model_trans,
            # model_pose,
            # model_betas,
            # model_delta_v,
            # occ_map,
            # img_path,
            # uv_flow,
        )

    def normalize_kp(self, kp, img_h, img_w, update_vis=False):
        vis = kp[:, 2, None] > 0
        new_kp = np.stack(
            [2 * (kp[:, 0] / img_w) - 1, 2 * (kp[:, 1] / img_h) - 1, kp[:, 2]]
        ).T

        if update_vis:
            new_kp[np.where(new_kp[:, 0] < -1), 2] = 0
            new_kp[np.where(new_kp[:, 0] > 1), 2] = 0
            new_kp[np.where(new_kp[:, 1] < -1), 2] = 0
            new_kp[np.where(new_kp[:, 1] > 1), 2] = 0

        new_kp = vis * new_kp

        return new_kp

    def get_camera_projection_matrix(self, f, c):
        P = np.hstack([np.eye(3), np.zeros((3, 1))])
        # Add camera matrix
        K = np.zeros((3, 3))
        K[0, 0] = f
        K[1, 1] = f
        K[2, 2] = 1
        K[0, 2] = c[0]
        K[1, 2] = c[1]
        KP = np.array(np.matrix(K) * np.matrix(P))
        return KP

    def my_project_points(self, ptsw, P):
        # Project world points ptsw(Nx3) into image points ptsi(Nx2) using the camera matrix P(3X4)
        nPts = ptsw.shape[0]
        ptswh = np.ones((nPts, 4))
        ptswh[:, :-1] = ptsw
        ptsih = np.dot(ptswh, P.T)
        ptsi = np.divide(ptsih[:, :-1], ptsih[:, -1][:, np.newaxis])
        return ptsi

    def my_anti_project_points(self, ptsi, P):
        nPts = ptsi.shape[0]
        ptsih = np.ones((nPts, 3))
        ptsih[:, :-1] = ptsi
        ptswh = np.dot(ptsih, np.array(np.matrix(P.T).I))
        nPts = ptswh.shape[0]
        if P[-1, -1] == 0:
            ptsw = ptswh[:, :-1]
        else:
            ptsw = np.divide(ptswh[:, :-1], ptswh[:, -1][:, np.newaxis])
        return ptsw

    def get_model_trans_for_cropped_image(self, trans, bbox, flength, img_w, img_h):
        """
        trans: 3  model translation
        bbox: 1 x 4 xmin, ymin, xmax, ymax
        flength: 1
        img_w: 1 width original image
        img_h: 1 height original image
        """
        # Location of the model in image frame (pixel coo)
        P = self.get_camera_projection_matrix(
            flength, np.array([img_w / 2.0, img_h / 2.0])
        )
        Q = np.zeros((1, 3))
        Q[0, :] = trans
        W = self.my_project_points(Q, P)

        # Location of the model w.r.t. the center of the bbox (pixel coo)
        E = np.zeros((1, 2))
        E[0, 0] = W[0, 0] - (bbox[0] + (bbox[2] - bbox[0]) / 2.0)
        E[0, 1] = W[0, 1] - (bbox[1] + (bbox[3] - bbox[1]) / 2.0)

        # Define the new camera for the bbox
        # Center of the bbox in the bbox frame
        c = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]]) / 2.0
        P = self.get_camera_projection_matrix(flength, c)
        P[-1, -1] = trans[2]
        # Location of the model in world space w.r.t. the new image and camera
        D = self.my_anti_project_points(E, P)

        trans[:] = np.array([D[0, 0], D[0, 1], trans[2]])

        return trans

    def crop_image(
        self, img, mask, bbox, kp, vis, camera_params, model_trans, occ_map, uv_flow
    ):
        img_orig_h, img_orig_w = img.shape[:2]

        # crop image and mask and translate kps
        img = image_utils.crop(img, bbox, bgval=self.opts["bgval"])
        if mask is not None:
            mask = image_utils.crop(mask, bbox, bgval=0)

        if occ_map is not None:
            occ_map = image_utils.crop(occ_map, bbox, bgval=0)

        if uv_flow is not None:
            # uv_flow has image coordinates in the first 2 channels and a mask in the third channel
            # image coordinates are normalized w.r.t the original size so their value is in [-1,1]
            # un-normalize uv_flow coordinates
            uv = uv_flow[:, :, :2]
            uv[:, :, 0] = uv[:, :, 0] * (img_orig_h / 2.0) + img_orig_h / 2.0
            uv[:, :, 1] = uv[:, :, 1] * (img_orig_w / 2.0) + img_orig_w / 2.0
            # Change the values
            uv[:, :, 0] -= bbox[0]
            uv[:, :, 1] -= bbox[1]
            img_h, img_w = img.shape[:2]
            uv_flow[:, :, 0] = (uv[:, :, 0] - (img_h / 2.0)) / (img_h / 2.0)
            uv_flow[:, :, 1] = (uv[:, :, 1] - (img_w / 2.0)) / (img_w / 2.0)

        if self.opts["translate_keypoints"] and kp is not None:
            kp[vis, 0] -= bbox[0]
            kp[vis, 1] -= bbox[1]

        if camera_params[0] > 0:
            if model_trans is not None:
                model_trans = self.get_model_trans_for_cropped_image(
                    model_trans, bbox, camera_params[0], img_orig_w, img_orig_h
                )
            camera_params[1][0] = img.shape[1] / 2.0
            camera_params[1][1] = img.shape[0] / 2.0

        return img, mask, kp, camera_params, model_trans, occ_map, uv_flow

    def scale_image(
        self, img, mask, kp, vis, camera_params, occ_map, orig_bbox, uv_flow
    ):
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[0]
        bheight = np.shape(img)[1]

        scale = self.opts["img_size"] / float(max(bwidth, bheight))
        img_scale, _ = image_utils.resize_img(img, scale)

        if occ_map is not None:
            occ_map, _ = image_utils.resize_img(occ_map, scale)

        if mask is not None:
            mask_scale, _ = image_utils.resize_img(mask, scale)
        else:
            mask_scale = None
        if kp is not None:
            kp[vis, :2] *= scale

        if uv_flow is not None:
            img_orig_h, img_orig_w = img.shape[:2]
            # un-normalize uv_flow coordinates
            uv = uv_flow[:, :, :2]
            uv[:, :, 0] = uv[:, :, 0] * (img_orig_h / 2.0) + img_orig_h / 2.0
            uv[:, :, 1] = uv[:, :, 1] * (img_orig_w / 2.0) + img_orig_w / 2.0
            # Change the values
            uv[:, :, 0] *= scale
            uv[:, :, 1] *= scale
            img_h, img_w = img_scale.shape[:2]
            uv_flow[:, :, 0] = (uv[:, :, 0] - (img_h / 2.0)) / (img_h / 2.0)
            uv_flow[:, :, 1] = (uv[:, :, 1] - (img_w / 2.0)) / (img_w / 2.0)

        bwidth = orig_bbox[2] - orig_bbox[0] + 1
        bheight = orig_bbox[3] - orig_bbox[1] + 1

        if camera_params[0] > 0:
            camera_params[0] *= scale
        camera_params[1] *= scale
        return img_scale, mask_scale, kp, camera_params, occ_map, uv_flow

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        (
            img,
            img_orig,
            kp,
            kp_norm,
            mask,
            camera_params,
            # texture_map,
            # model_trans,
            # model_pose,
            # model_betas,
            # model_delta_v,
            # _,
            # img_path,
            # uv_flow,
        ) = self.forward_img(index)

        camera_params[0].shape = 1

        elem = {
            "img": img,
            "img_orig": img_orig,
            "inds": index,
            # "img_path": img_path,
        }
        if kp is not None:
            elem["kp_raw"] = kp
        if kp_norm is not None:
            elem["keypoints"] = kp_norm
        if mask is not None:
            elem["mask"] = mask
        # if texture_map is not None:
        #     elem["texture_map"] = texture_map
        # elem["camera_params_c"] = camera_params[1]
        # if camera_params[0]:
        #     elem["camera_params"] = np.concatenate(camera_params)
        # if model_trans is not None:
        #     elem["model_trans"] = model_trans
        # if model_pose is not None:
        #     elem["model_pose"] = model_pose
        # if model_betas is not None:
        #     elem["model_betas"] = model_betas
        # if model_delta_v is not None:
        #     elem["model_delta_v"] = model_delta_v
        # if uv_flow is not None:
        #     elem["uv_flow"] = uv_flow

        if self.filter_key is not None:
            if self.filter_key not in elem.keys():
                print("Bad filter key %s" % self.filter_key)
            if self.filter_key == "camera_params":
                # Return both vis and sfm_pose
                vis = elem["kp"][:, 2]
                elem = {
                    "vis": vis,
                    "camera_params": elem["camera_params"],
                }
            else:
                elem = elem[self.filter_key]

        return elem


# ------------ Data Loader ----------- #
# ------------------------------------ #
def base_loader(d_set_func, opts, filter_key=None, shuffle=True, filter_name=None):
    dset = d_set_func(opts, filter_key=filter_key, filter_name=filter_name)
    return DataLoader(
        dset,
        batch_size=opts["batch_size"],
        shuffle=shuffle,
        num_workers=0,  # TODO: num_workers=opts["n_data_workers"],
        drop_last=True,
    )
