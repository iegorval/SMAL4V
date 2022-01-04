"""
Evaluation on the test set.

Original author: Silvia Zuffi (https://github.com/silviazuffi/smalst)
Refactoring and Python 3 porting: Valeria Iegorova (https://github.com/iegorval/smalmv)
MIT licence
"""
import os
import cv2
import json
import shutil
import numpy as np
import pickle as pkl
import torch
import imageio
import scipy.io as sio
import matplotlib.pyplot as plt
from src import smal_predictor as pred_util
from utils import image as img_util
from glob import glob
from typing import Tuple, Dict, List, Any, Optional
from src.smal_predictor import MeshPredictor
from utils.smal_vis import VisRenderer
from src import loss_utils


# TODO: pass as flags
CONFIG_DIR = "configs/"
CONFIG_EVAL = "evaluation_params_cows_smbld.json"
CONFIG_DATA = "cows_data.json"
KP_COLOR = (255, 0, 0)


def load_input_with_bbox(
    img_path: str, kp: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the input RGB image and the segmentation mask. Crops both of them ...
    Args:
        img_path: path to the input RGB image.
        kp: the keypoints, corresponding to the input image.
    Returns:
        img: cropped input RGB image.
        mask_image: corresponding segmentation mask.
        kp: re-centered keypoints, corresponding to the input image.
    """
    # Load the mask
    mask_img = imageio.imread(img_path.replace("images", "bgsub")) / 255.0
    # Load the image
    img = imageio.imread(img_path) / 255.0
    where = np.array(np.where(mask_img))
    xmin, ymin, _ = np.amin(where, axis=1)
    xmax, ymax, _ = np.amax(where, axis=1)
    mask_img = mask_img[xmin:xmax, ymin:ymax, :]
    img = img[xmin:xmax, ymin:ymax, :]
    kp[:, 0] = kp[:, 0] - ymin
    kp[:, 1] = kp[:, 1] - xmin
    return img, mask_img, kp


def read_full_image(
    img_path: str,
    img_name: str,
    img_ext: str,
    bboxes: Dict[str, List[int]],
    scale_factor: float,
    img_size: int = 256,
) -> np.ndarray:
    """
    If the image is to be padded with the original image, the full-size image is
    read. Then, it is cropped and resized to the size, corresponding to the size
    of the mesh predictor's input.
    Args:
        img_path: path to the input RGB image.
        img_name: name of the input RGB image.
        img_ext: extension of the input RGB image.
        bboxes: pre-computed bounding boxes.
        scale_factor: scale for the resizing of the input image.
        img_size: image size, on which the network was trained.
    Returns:
        img: resized and cropped image.
    """
    if os.path.exists(img_path):
        full_img = imageio.imread(img_path) / 255.0
        bbox_orig = np.array(bboxes[f"{img_name}{img_ext}"])
        # sf = img_in_shape[0] / (1.0 * bbox_orig[3])
        # print("SF", sf, "SCALE FACTOR", scale_factor)
        new_img, _ = img_util.resize_img(full_img, scale_factor)
        center = np.round(
            np.asarray(
                [
                    (bbox_orig[2] / 2.0 + bbox_orig[0]) * scale_factor,
                    (bbox_orig[3] / 2.0 + bbox_orig[1]) * scale_factor,
                ]
            )
        ).astype(int)
        # new_img, _ = img_util.resize_img(full_img, sf * scale_factor)
        # center[0] = np.round(
        #     (bbox_orig[2] / 2.0 + bbox_orig[0]) * sf * scale_factor
        # ).astype(int)
        # center[1] = np.round(
        #     (bbox_orig[3] / 2.0 + bbox_orig[1]) * sf * scale_factor
        # ).astype(int)
        bbox2 = np.hstack([center - img_size / 2.0, center + img_size / 2.0])
        img = img_util.crop(new_img, bbox2, bgval=0)
    return img


def mirror_keypoints(
    kp: np.ndarray, vis: np.ndarray, img_width: int, dx2sx: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    exchange keypoints from left to right and update x value
    names = ['leftEye','rightEye','chin','frontLeftFoot','frontRightFoot',
             'backLeftFoot','backRightFoot','tailStart','frontLeftKnee',
             'frontRightKnee','backLeftKnee','backRightKnee','leftShoulder',
             'rightShoulder','frontLeftAnkle','frontRightAnkle','backLeftAnkle'
             'backRightAnkle','neck','TailTip','leftEar','rightEar',
             'nostrilLeft','nostrilRight','mouthLeft','mouthRight',
             'cheekLeft','cheekRight']
    Args:
        kp: the keypoints, corresponding to the input image.
        vis: visibility flags, corresponding to the keypoints.
        img_width: the width of the input image.
        dx2sx:

    Returns:
        kp_m: mirrored keypoints.
        vis_m: mirrored visibility flags.
    """
    kp_m = np.zeros_like(kp)
    kp_m[:, 0] = img_width - kp[dx2sx, 0] - 1
    kp_m[:, 1] = kp[dx2sx, 1]
    vis_m = np.zeros_like(vis)
    vis_m[:] = vis[dx2sx]
    return kp_m, vis_m


def mirror_image(img: np.ndarray) -> np.ndarray:
    """
    Mirrors the input image to increase the dataset size.
    Args:
        img: input RGB image.
    Returns:
        img_m: mirrored image.
    """
    if len(img.shape) == 3:
        img_m = img[:, :, ::-1].copy()
    else:
        img_m = img[:, ::-1].copy()
    return img_m


def visualize_opt(img, predictor, renderer, data, out_path):
    """
    Visualizes the results of the per-instance model optimization.
    Args:
        img: input RGB image.
        predictor: trained model, predicting a 3D mesh from the input image.
        renderer: renderer, which projects the optimized model into a 2D image.
        data:
        out_path: path, where the renderings are saved.
    """
    pose = torch.Tensor(data["pose"]).cuda(0)
    trans = torch.Tensor(data["trans"]).cuda(0)
    del_v = torch.Tensor(data["delta_v"]).cuda(0)
    vert = predictor.model.get_smal_verts(pose, None, trans, del_v)
    cam = 128 * np.ones((3))
    cam[0] = data["scale"][0, :]
    cam = torch.Tensor(cam).cuda(0)
    shape_pred = renderer(vert, cam)
    img = np.transpose(img, (1, 2, 0))
    I = 0.3 * 255 * img + 0.7 * shape_pred
    imageio.imwrite(out_path, I)


def visualize(
    img: np.ndarray,
    outputs: Dict[str, torch.Tensor],
    renderer: VisRenderer,  # TODO: should be in class
    opts: Dict[str, Any],
    kp_gt: Optional[np.ndarray],
    kp_pred: np.ndarray,  # TODO: remove (is in outputs)
    vis: np.ndarray,
    out_name: str,
    tex_mask: imageio.core.util.Array,
    show_plt: bool = False,
):
    """
    Visualizes the result of the model prediction for the given image.
    Args:
        img: input RGB image.
        outputs: dictionary, containing the model predictions.
        renderer: renderer, which projects the predicted model into a 2D image.
        opts: the dictionary of the evaluation parameters.
        kp_gt: the ground-truth keypoints (if available).
        kp_pred: the predicted keypoints.
        vis: visibility flags, corresponding to the keypoints.
        out_name: name of the output rendered image.
        tex_mask:
        show_plt: whether to show the rendered prediction results.
    """
    img_pred = None
    vert = outputs["verts"][0]
    cam = outputs["cam_pred"][0]
    mask = outputs["mask_pred"].cpu().detach().numpy()

    if "texture" in outputs.keys():
        texture = outputs["texture"][0]
        uv_image = outputs["uv_image"][0].cpu().detach().numpy()
        T = uv_image.transpose(1, 2, 0)
        img_pred = np.flipud(
            renderer(vert, opts["gpu_id"], cams=cam, texture=texture)
        ).astype(np.uint8)
    # if "occ_pred" in outputs.keys():
    #    occ_pred = outputs["occ_pred"][0].cpu().detach().numpy()

    shape_pred = renderer(vert, opts["gpu_id"], cams=cam).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    mask = np.transpose(mask, (1, 2, 0))[:, :, 0]

    Iov = (0.3 * 255 * img + 0.7 * shape_pred).astype(np.uint8)
    imageio.imwrite(os.path.join(opts["evaluation_path"], "shape_ov_" + out_name), Iov)
    imageio.imwrite(
        os.path.join(opts["evaluation_path"], "shape_" + out_name), shape_pred
    )

    if "texture" in outputs.keys() and img_pred is not None:
        N = 0.6 * np.abs(tex_mask[:, :, :3] - 1) * 255
        texture_map = (N + T * tex_mask[:, :, :3] * 255).astype(np.uint8)
        imageio.imwrite(
            os.path.join(opts["evaluation_path"], "tex_" + out_name), texture_map
        )
        imageio.imwrite(
            os.path.join(opts["evaluation_path"], "img_" + out_name), img_pred
        )

    if show_plt:
        _, ax = plt.subplots(2, 3)
        ax[0][0].imshow(img)
        ax[0][0].set_title("input")
        ax[0][0].axis("off")
        ax[0][1].imshow(img)
        ax[0][1].imshow(shape_pred, alpha=0.7)
        if kp_gt is not None:
            idx = np.where(vis)
            ax[0][1].scatter(kp_gt[idx, 0], kp_gt[idx, 1], c="k")
            ax[0][1].scatter(kp_pred[idx, 0], kp_pred[idx, 1], c="r")
        ax[0][1].set_title("pred mesh")
        ax[0][1].axis("off")
        ax[0][2].imshow(mask, cmap="gray")
        ax[0][2].axis("off")
        if "texture" in outputs.keys():
            ax[1][0].imshow(img_pred)
            ax[1][0].set_title("pred mesh w/texture")
            ax[1][0].axis("off")
            ax[1][1].imshow(T)
            ax[1][1].axis("off")
            ax[1][2].imshow(T * tex_mask[:, :, :3])
            ax[1][2].axis("off")
        out_path = os.path.join(opts["visualizations_dir"], out_name)
        plt.savefig(out_path, bbox_inches="tight")


def save_params(idx: int, outputs: Dict[str, torch.Tensor]):
    """
    Serialize the dictionary of the model parameters.
    Args:
        idx: index of the currently processed image.
        outputs: dictionary, containing the model predictions.
    """
    data = {
        "pose": outputs["pose_pred"].data.detach().cpu().numpy()[0, :],
        "verts": outputs["verts"].data.detach().cpu().numpy()[0, :],
        "f": outputs["f"].data.detach().cpu().numpy()[0, :],
        "v": outputs["v"].data.detach().cpu().numpy()[0, :],
    }
    pkl.dump(data, open("data_" + str(idx) + ".pkl", "wb"))


def compute_iou_and_overlap(
    mask_gt: np.ndarray, mask_pred: np.ndarray, opts: Dict[str, Any]
) -> Tuple[float, float]:
    """
    Computes the Intersection-over-Union and the overlap between the predicted and ground
    truth segmentation masks.
    Args:
        mask_gt: the ground truth segmentation mask.
        mask_pred: the predicted segmentation mask.
        opts: the dictionary of the evaluation parameters.
    Returns:
        iou: the Intersection-over-Union between the ground truth mask and the predicted masks.
        overlap: the overlap between the ground truth mask and the predicted mask.
    """
    iou, overlap = None, None
    if opts["segm_eval"]:
        overlap = np.sum(mask_gt * mask_pred) / (np.sum(mask_gt) + np.sum(mask_pred))
        iou = np.sum(mask_gt * mask_pred) / (
            np.sum(mask_gt) + np.sum(mask_pred) - np.sum(mask_gt * mask_pred)
        )
    return iou, overlap


def load_keypoints(
    img_path: str, opts: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads either the synthetic or the user-clicked keypoints and their
    corresponding visibility flags.
    Args:
        img_path: path to the input RGB image.
        opts: the dictionary of the evaluation parameters.
    Returns:
        kp: the keypoints, corresponding to the input image.
        vis: visibility flags, corresponding to the keypoints.
    """
    kp, vis = None, None
    if opts["use_annotations"]:
        if opts["synthetic"]:
            anno_path = os.path.join(
                opts["anno_path"],
                os.path.basename(img_path).replace(opts["img_ext"], ".pkl"),
            )
            res = pkl.load(open(anno_path, "rb"), encoding="latin1")
            kp = res["keypoints"]
            vis = np.ones((kp.shape[0], 1), dtype=bool)
        elif opts["kp_ext"] == ".pkl":
            fbase = os.path.join(
                opts["anno_path"], img_path.split(".")[0].split("/")[-1]
            )
            kp_vis = pkl.load(open(f"{fbase}.pkl", "rb"))
            kp = kp_vis[:, :2]
            vis = kp_vis[:, 2].reshape((-1, 1))
        elif opts["kp_ext"] == ".mat":
            anno_path = os.path.join(
                opts["anno_path"],
                os.path.basename(img_path).replace(
                    opts["img_ext"], "_ferrari-tail-face.mat"
                ),
            )
            res = sio.loadmat(anno_path, squeeze_me=True, struct_as_record=False)
            res = res["annotation"]
            kp = res.kp.astype(float)
            invisible = res.invisible
            vis = np.atleast_2d(~invisible.astype(bool)).T
        else:
            raise "Unsuppored Keypoints Format"
    return kp, vis


def get_image_with_annotations(
    img_path: str, kp: np.ndarray, bboxes: Dict[str, List[int]], opts: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For the synthetic images, loads the input image and the corresponding
    segmentation mask, both
    Args:
        img_path: path to the input RGB image.
        kp: the keypoints, corresponding to the input image.
        bboxes: pre-computed bounding boxes.
        opts: the dictionary of the evaluation parameters.
    Returns:
        img:
        kp: the keypoints, corresponding to the input image.
    """
    img = imageio.imread(img_path) / 255.0
    # img = img[:, :, :3]  TODO: any pngs?
    scale_factor = float(opts["img_size"] - 2 * opts["border"]) / np.max(img.shape[:2])
    img, _ = img_util.resize_img(img, scale_factor)
    img, _, bbox = img_util.crop_image(
        img, img_size=opts["img_size"], bgval=opts["bgval"]
    )

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    if opts["bgval"] == -1:
        # Replace the border with the real background
        full_img_path = os.path.join(
            os.path.dirname(img_path), "full_size", img_name + "*" + opts["img_ext"]
        )
        full_img_path = glob(full_img_path)[0]
        img = read_full_image(
            img_path=full_img_path,
            img_name=img_name,
            img_ext=opts["img_ext"],
            bboxes=bboxes,
            scale_factor=scale_factor,
            img_size=opts["img_size"],
        )

    img, _ = img_util.resize_img(img, 256 / 257.0)
    img = np.transpose(img, (2, 0, 1))

    if kp is not None:
        kp = kp * scale_factor
        kp[:, 0] -= bbox[0]
        kp[:, 1] -= bbox[1]

    if opts["segm_eval"]:  # TODO: unify options names
        mask_path = os.path.join(
            opts["img_path"],
            "masks",
            os.path.basename(img_path).replace("jpg", "png"),
        )
        mask = read_full_image(
            img_path=mask_path,
            img_name=img_name,
            img_ext=opts["img_ext"],
            bboxes=bboxes,
            scale_factor=scale_factor,
            img_size=opts["img_size"],
        )
        mask, _ = img_util.resize_img(mask, 256 / 257.0)
        mask = mask[:, :, 0]

    return img, kp, mask


def save_input(
    img_path: str,
    img: np.ndarray,
    opts: Dict[str, Any],
    kp: Optional[np.ndarray] = None,
    vis: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    mirrored: bool = False,
):
    """
    Save the preprocessed RGB input and the corresponding segmentation mask if
    requested by the parameters from opts. For the mirrored image, save only
    the image itself, without the corresponding segmentation mask.
    Args:
        img_path: path to the input RGB image.
        img: RGB image, processed by the mesh predictor.
        kp:
        vis:
        mask: the corresponding segmentation mask.
        opts: the dictionary of the evaluation parameters.
        mirrored: whether the mirrored input version is being saved.
    """
    label = "proc_mirr" if mirrored else "proc"

    if opts["save_processed_input"]:
        img_with_kp = (255 * np.transpose(img, (1, 2, 0))).astype(np.uint8)
        for i in range(kp.shape[0]):
            if vis[i] == 0:
                continue
            img_with_kp = cv2.circle(
                img_with_kp, (int(kp[i, 0]), int(kp[i, 1])), 4, KP_COLOR, -1
            )
        img_name = f"{label}_" + os.path.basename(img_path)
        img_path_out = os.path.join(opts["processed_input_path"], img_name)
        imageio.imwrite(img_path_out, img_with_kp)

        if opts["segm_eval"] and not mirrored:  # TODO: why not mirrored?
            mask = (255 * mask).astype(np.uint8)
            mask_name = f"mask_{label}_" + os.path.basename(img_path)
            mask_path_out = os.path.join(opts["processed_input_path"], mask_name)
            imageio.imwrite(mask_path_out, mask)


def get_pipeline_predictions(
    idx: int,
    img,
    predictor: MeshPredictor,
    shape_f: np.ndarray,
    code: str,
    global_rotation: int,
    opts: Dict[str, Any],
    label: str = "proc",
):  # TODO: output types (data?)
    """

    Args:
        idx: index of the currently processed image.
        img:
        predictor: trained model, predicting a 3D mesh from the input image.
        shape_f:
        code: identifier of the evaluated image.
        global_rotation:
        opts: the dictionary of the evaluation parameters.
    Returns:

    """
    data, outputs = None, None
    if opts["test_optimization_results"]:
        res_file = os.path.join(
            opts["optimization_dir"], f"{label}_" + code + "_best_res.pkl"
        )
        mask_file = os.path.join(
            opts["optimization_dir"], f"{label}_" + code + "_best_mask.png"
        )
        if not os.path.exists(res_file) or not os.path.exists(mask_file):
            res_file = os.path.join(
                opts["optimization_dir"], f"{label}_" + code + "_init_res.pkl"
            )
            mask_file = os.path.join(
                opts["optimization_dir"], f"{label}_" + code + "_init_mask.png"
            )
        else:
            print("found optimization result")
        data = pkl.load(open(res_file, "rb"), encoding="latin1")
        mask_pred = imageio.imread(mask_file) / 255.0
    else:
        batch = {"img": torch.Tensor(np.expand_dims(img, 0))}
        outputs = predictor.predict(batch, rot=global_rotation)
        mask_pred = outputs["mask_pred"].detach().cpu().numpy()[0, :, :]
        if opts["use_shape_predictor"]:
            shape_f[idx, :] = outputs["shape_f"].detach().cpu().numpy()
    return data, mask_pred, outputs


def compute_kp_error(
    idx: int,
    kp: np.ndarray,
    vis: np.ndarray,
    data: Optional[Any],  # CHECK THE DATATYPE
    outputs: Dict[str, torch.Tensor],
    alpha: Tuple[float, ...],
    err_tot: np.ndarray,
    opts: Dict[str, Any],
) -> np.ndarray:
    """
    Computers the keypoints error as the distance the predicted and annotated keypoints.
    Args:
        idx: index of the currently processed image.
        kp: the keypoints, corresponding to the input image.
        vis: visibility flags, corresponding to the keypoints.
        data: kp
        outputs: dictionary, containing the model predictions.
        alpha: thresholds for the keypoints error computation.
        err_tot: all errors for all images and all alpha thresholds.
        opts: the dictionary of the evaluation parameters.
    Returns:
        kp_pred: predicted keypoints.
    """
    kp_pred = None
    if opts["use_annotations"]:
        if opts["test_optimization_results"]:
            kp_pred = data["kp_pred"][0, :, :]
        else:
            kp_pred = outputs["kp_pred"].cpu().detach().numpy()[0, :, :]
        if opts["pipeline_version"] == "smalst":
            kp_pred = (kp_pred + 1.0) * 128
        # TODO: invert x-y axis for WLDO dataset
        kp_pred = kp_pred.astype(int)
        kp_pred_torch = torch.Tensor(kp_pred.reshape((1, 30, 2)))
        kp_vis = torch.Tensor(np.hstack([kp, vis]).reshape((1, 30, 3)))
        kp_loss = loss_utils.kp_l1_loss(kp_pred_torch, kp_vis)
        print("Keypoints loss:", kp_loss)

        # TODO: normalize by silhouette area?
        kp_diffs = np.linalg.norm(kp_pred - kp, axis=1)[np.where(vis == 1)[0]]
        for a in range(len(alpha)):
            bound = opts["img_size"] * alpha[a]
            kp_err = np.mean(kp_diffs < bound)
            print(
                f"* PCK {round(kp_err * 100, 2)}% at threshold {alpha[a]} ({round(bound, 2)} px)"
            )
            err_tot[idx, a] = kp_err
    return kp_pred


def visualize_results(
    idx: int,
    img: np.ndarray,
    outputs: Dict[str, torch.Tensor],
    predictor: MeshPredictor,
    data: Optional[Any],  # CHECK THE DATATYPE
    kp_gt: np.ndarray,
    kp_pred: np.ndarray,
    vis: np.ndarray,
    tex_mask: imageio.core.util.Array,
    opts: Dict[str, Any],
):
    """
    Depending on the evaluation parameters, visualizes either the feed-forward
    prediction or the per-instance optimization results.
    Args:
        idx: index of the currently processed image.
        img: input RGB image.
        outputs: dictionary, containing the model predictions.
        predictor: trained model, predicting a 3D mesh from the input image.
        data:
        kp_gt: ground-truth keypoints.
        kp_pred: predicted keypoints of the SMAL model.
        vis: visibility flags, corresponding to the keypoints.
        tex_mask:
        opts: the dictionary of the evaluation parameters.
    """
    if opts["show"] and not opts["test_optimization_results"]:  # opts["visualize"]?
        renderer = predictor.vis_rend
        renderer.set_light_dir([0, 1, -1], 0.4)  # was not in mirrored?
        print("INDEX:", idx)
        visualize(
            img,
            outputs,
            predictor.vis_rend,
            opts,
            kp_gt=kp_gt,  # WHY IS IT ALWAYS NONE?
            kp_pred=kp_pred,
            vis=vis,
            out_name=opts["name"] + "_test_%03d" % idx + ".png",
            tex_mask=tex_mask,
            show_plt=opts["visualize"],
        )
    if opts["show"] and opts["test_optimization_results"]:
        visualize_opt(
            img,
            predictor,
            predictor.vis_rend,
            data,
            opts["visualizations_dir"] + "_opt_%03d" % idx + ".png",
        )


def evaluate_single_image(
    idx: int,
    img: np.ndarray,
    mask: np.ndarray,
    tex_mask: imageio.core.util.Array,
    kp: np.ndarray,
    vis: np.ndarray,
    alpha: Tuple[float, ...],
    err_tot: np.ndarray,
    predictor: MeshPredictor,
    shape_f: np.ndarray,
    img_name: str,
    rotation: int,
    opts: Dict[str, Any],
    mirrored: bool = False,
) -> Tuple[float, float]:
    """
    Evaluates the model performance on a single input image. Computes the IOU,
    the overlap and the keypoint error. Then, visualizes and saves the results.
    Args:
        idx: index of the currently processed image.
        img: input RGB image.
        mask_gt: the ground truth segmentation mask.
        tex_mask:
        kp: the keypoints, corresponding to the input image.
        vis: visibility flags, corresponding to the keypoints.
        alpha: thresholds for the keypoints error computation.
        err_tot: all errors for all images and all alpha thresholds.
        predictor: trained model, predicting a 3D mesh from the input image.
        shape_f:
        img_name: identifier of the evaluated image.
        rotation:
        opts: the dictionary of the evaluation parameters.
        mirrored: whether the mirrored input version is being processed.
    """
    label = "proc_mirr" if mirrored else "proc"
    data, mask_pred, outputs = get_pipeline_predictions(
        idx, img, predictor, shape_f, img_name, rotation, opts, label=label
    )
    iou, overlap = compute_iou_and_overlap(mask, mask_pred, opts)
    kp_pred = compute_kp_error(idx, kp, vis, data, outputs, alpha, err_tot, opts)
    visualize_results(
        idx, img, outputs, predictor, data, kp, kp_pred, vis, tex_mask, opts
    )
    print("Overlap", overlap)
    print("Intersection-over-Union:", iou)
    return iou, overlap


def reset_output_dirs(opts: Dict[str, Any]):
    if os.path.exists(opts["visualizations_dir"]):
        shutil.rmtree(opts["visualizations_dir"])
    os.makedirs(opts["visualizations_dir"])
    if os.path.exists(opts["processed_input_path"]):
        shutil.rmtree(opts["processed_input_path"])
    os.makedirs(opts["processed_input_path"])


def evaluate_all_images(
    n_images: int,
    image_path_list: List[str],
    alpha: Tuple[float, ...],
    predictor: MeshPredictor,
    tex_mask: imageio.core.util.Array,
    bboxes: Dict[str, List[int]],
    opts: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates all the images, specified by the list of their corresponding paths
    image_path_list. For each of them, predicts the 3D mesh with the pre-trained
    model, computes the resulting errors and saves the result.
    Args:
        n_images: number of the evaluation images.
        image_path_list: paths to all the evaluation images.
        alpha: thresholds for the keypoints error computation.
        predictor: trained model, predicting a 3D mesh from the input image.
        tex_mask:
        bboxes: pre-computed bounding boxes.
        opts: the dictionary of the evaluation parameters.
    Returns:
        err_tot: all errors for all images and all alpha thresholds.
        overlap_tot: overlap for each evaluated image.
        iou_tot: Intersection-over-Union for each evaluated image.
    """
    idx = 0
    global_rotation, mirr_global_rotation = 0, 0
    shape_f = np.zeros((n_images, 40))
    overlap_tot = np.zeros(n_images)
    iou_tot = np.zeros(n_images)
    err_tot = np.zeros((n_images, len(alpha)))

    reset_output_dirs(opts)
    for (_, img_path) in enumerate(image_path_list):
        print(img_path)
        kp, vis = load_keypoints(img_path, opts)
        img, kp, mask = get_image_with_annotations(img_path, kp, bboxes, opts)
        save_input(img_path, img, opts, kp, vis, mask)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        iou, overlap = evaluate_single_image(
            idx=idx,
            img=img,
            mask=mask,
            tex_mask=tex_mask,
            kp=kp,
            vis=vis,
            alpha=alpha,
            err_tot=err_tot,
            predictor=predictor,
            shape_f=shape_f,
            img_name=img_name,
            rotation=global_rotation,
            opts=opts,
        )
        iou_tot[idx], overlap_tot[idx] = iou, overlap
        idx += 1
        # if opts["mirror"]:
        #     img_m, mask_gt_m = mirror_image(img), mirror_image(mask_img)
        #     img_width = (
        #         mask_img.shape[1] if opts["test_optimization_results"] else img.shape[2]
        #     )
        #     kp_m, vis_m = mirror_keypoints(kp, vis, img_width, dx2sx=opts["dx2sx"])
        #     save_input(img_path, img_m, opts, kp, mirrored=True)
        #     evaluate_single_image(
        #         idx=idx,
        #         img=img_m,
        #         mask_gt=mask_gt_m,
        #         tex_mask=tex_mask,
        #         kp=kp_m,
        #         vis=vis_m,
        #         alpha=alpha,
        #         err_tot=err_tot,
        #         predictor=predictor,
        #         shape_f=shape_f,
        #         code=code,
        #         rotation=mirr_global_rotation,
        #         opts=opts,
        #         mirrored=True,
        #     )
        #     idx += 1
    return err_tot, overlap_tot, iou_tot


def main(opts):
    tex_mask = imageio.imread(opts["texture_mask_path"]) / 255.0
    bboxes = json.load(open(opts["bboxes_path"]))
    os.makedirs(opts["evaluation_path"], exist_ok=True)  # TODO: move somewhere else
    image_paths = sorted(glob(opts["img_path"] + "*" + opts["img_ext"]))
    n_images = 2 * len(image_paths) if opts["mirror"] else len(image_paths)
    print("Test set size:", n_images)
    predictor = pred_util.MeshPredictor(opts) if opts["show"] else None
    print(predictor)
    alpha = (0.01, 0.02, 0.05, 0.1, 0.15)
    err_tot, overlap, iou = evaluate_all_images(
        n_images, image_paths, alpha, predictor, tex_mask, bboxes, opts
    )
    if opts["use_annotations"]:
        print("Total PCK")
        print(f"{np.mean(err_tot, axis=0)} +- {np.std(err_tot, axis=0)}")
        print("Total Overlap")
        print(f"{np.mean(overlap, axis=0)} +- {np.std(overlap, axis=0)}")
        print("Total IOU")
        print(f"{np.mean(iou, axis=0)} +- {np.std(iou, axis=0)}")


if __name__ == "__main__":
    fname_data = os.path.join(CONFIG_DIR, CONFIG_DATA)
    with open(fname_data) as f:
        opts_data = json.load(f)
    fname_eval = os.path.join(CONFIG_DIR, CONFIG_EVAL)
    with open(fname_eval) as f:
        opts_eval = json.load(f)
    opts = opts_data | opts_eval
    main(opts)
