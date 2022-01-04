import os
import cv2
import scipy.io
import imageio
import numpy as np
import pickle as pkl
from glob import glob
from typing import Tuple, Dict, List, Any, Optional
from scipy.ndimage import distance_transform_edt


def get_image_with_annotations(
    img_path: str,
    kp: np.ndarray,
    bboxes: Dict[str, List[int]],
    opts: Dict[str, Any],
    masks_dir: Optional[str] = None,
    img_cropped: Optional[bool] = True,
    mask_cropped: Optional[bool] = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For the synthetic images, loads the input image and the corresponding
    segmentation mask, both
    Args:
        img_path: path to the input RGB image.
        kp: the keypoints, corresponding to the input image.
        bboxes: pre-computed bounding boxes.
        opts: the dictionary of the evaluation parameters.
        mask_dir:
    Returns:
        img:
        kp: the keypoints, corresponding to the input image.
        mask:
    """
    # TODO: stop reading image twice like Silvia
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    if img_cropped:  # for compatibility with Silvia's version
        img = imageio.imread(img_path) / 255.0
        img_in_shape = img.shape
        max_side = np.max(img.shape[:2])
        scale_factor = float(opts["img_size"] - 2 * opts["border"]) / max_side
        img, _ = resize_img(img, scale_factor)
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)[::-1]
        # TODO: why there is crop()? is this Silvia's synthetic data format?
        # img, _, bbox = crop_image(img, img_size=opts["img_size"], bgval=opts["bgval"])
        full_img_path = os.path.join(
            os.path.dirname(img_path),
            "full_size",
            img_name + "*" + opts["img_ext"],
        )
        full_img_path = glob(full_img_path)[0]
    else:
        full_img_path = img_path
        scale_factor = None
        img_in_shape = None

    # TODO: unify bboxes keys
    bboxes_key = img_name + opts["img_ext"]
    if bboxes_key not in bboxes:
        bboxes_key = img_name

    if opts["bgval"] == -1 or not img_cropped:
        max_width = opts["max_img_width"] if "max_img_width" in opts else None
        img, bbox, scale_factor = read_full_image(
            img_path=full_img_path,
            img_in_shape=img_in_shape,
            bbox=bboxes[bboxes_key],
            scale_factor=scale_factor,
            img_size=opts["img_size"],
            max_width=max_width,
            border=opts["border"],
        )

    img, _ = resize_img(img, 256 / 257.0)
    img = np.transpose(img, (2, 0, 1))

    if img_cropped:  # compatibility with Silvia's version
        bbox = np.hstack(
            [center - opts["img_size"] / 2.0, center + opts["img_size"] / 2.0]
        )

    if kp is not None:
        kp = kp * scale_factor
        kp[:, 0] -= bbox[0]
        kp[:, 1] -= bbox[1]

    mask = None
    if opts["use_masks"]:
        mask_name = os.path.basename(img_path).replace("jpg", "png")
        if masks_dir is None:
            if "masks_path" not in opts:
                mask_path = os.path.join(
                    opts["img_path"],
                    "masks",
                    mask_name,
                )
            else:
                mask_path = os.path.join(opts["masks_path"], mask_name)
        else:
            mask_path = os.path.join(masks_dir, mask_name)

        if os.path.exists(mask_path):
            if mask_cropped:
                mask = imageio.imread(mask_path) / 255.0
                mask, _ = resize_img(mask, scale_factor)
                mask = crop(mask, bbox, bgval=0)
            else:
                mask, _, _ = read_full_image(
                    img_path=mask_path,
                    # img_name=img_name,
                    # img_ext=opts["img_ext"],
                    bbox=bboxes[bboxes_key],
                    # scale_factor=scale_factor,
                    scale_factor=None,
                    img_size=opts["img_size"],
                    max_width=max_width,
                    border=opts["border"],
                )
            mask, _ = resize_img(mask, 256 / 257.0)
            if len(mask.shape) > 2:
                mask = mask[:, :, 0]  # TODO: unify

    return img, kp, mask


def load_keypoints(
    img_path: str, opts: Dict[str, Any], keypoints_dir: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads either the synthetic or the user-clicked keypoints and their
    corresponding visibility flags.
    Args:
        img_path: path to the input RGB image.
        opts: the dictionary of the evaluation parameters.
        keypoints_dir:
    Returns:
        kp: the keypoints, corresponding to the input image.
        vis: visibility flags, corresponding to the keypoints.
    """
    kp, vis = None, None
    if opts["use_keypoints"]:
        if keypoints_dir is None:
            keypoints_dir = opts["anno_path"]
        if opts["synthetic"]:  # Silvia's synthetic zebra dataset format
            anno_path = os.path.join(
                keypoints_dir,
                os.path.basename(img_path).replace(opts["img_ext"], ".pkl"),
            )
            res = pkl.load(open(anno_path, "rb"), encoding="latin1")
            kp = res["keypoints"]
            vis = np.ones((kp.shape[0], 1), dtype=bool)
        elif opts["kp_ext"] == ".pkl":  # AnimalKey dataset format
            keypoints_name = os.path.split(img_path)[-1].split(".")[0] + ".pkl"
            keypoints_path = os.path.join(keypoints_dir, keypoints_name)
            if os.path.exists(keypoints_path):
                kp_vis = pkl.load(open(keypoints_path, "rb"))
                kp = kp_vis[:, :2]
                vis = kp_vis[:, 2].reshape((-1, 1))
            else:
                kp, vis = None, None
        elif opts["kp_ext"] == ".mat":  # Silvia's real zebra dataset format
            anno_path = os.path.join(
                keypoints_dir,
                os.path.basename(img_path).replace(
                    opts["img_ext"], "_ferrari-tail-face.mat"
                ),
            )
            res = scipy.io.loadmat(anno_path, squeeze_me=True, struct_as_record=False)
            res = res["annotation"]
            kp = res.kp.astype(float)
            invisible = res.invisible
            vis = np.atleast_2d(~invisible.astype(bool)).T
        else:
            raise "Unsuppored Keypoints Format"
    return kp, vis


def read_full_image(
    img_path: str,
    # img_name: str,
    # img_ext: str,
    # bboxes: Dict[str, List[int]],
    bbox,
    scale_factor: float,
    img_size: int = 256,
    max_width: int = None,
    img_in_shape: Optional[Any] = None,
    border: int = 0,  # TODO: maybe pass opts
) -> np.ndarray:  # TODO: rename
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
        max_width:
    Returns:
        img: resized and cropped image.
    """
    if os.path.exists(img_path):
        full_img = imageio.imread(img_path) / 255.0
        if max_width is not None:
            max_height = full_img.shape[0] * (max_width / full_img.shape[1])
            bbox = np.asarray(bbox) * (max_width / full_img.shape[1])
            width = int(min(max_width, full_img.shape[1]))
            height = int(min(max_height, full_img.shape[0]))
            full_img = cv2.resize(full_img, (width, height))
            sf = 1.0
        else:  # for compatibility with Silvia's code
            sf = img_in_shape[0] / (1.0 * bbox[3])

        if scale_factor is None:
            scale_factor = float(img_size - 2 * border) / max(bbox[2:])

        new_img, _ = resize_img(full_img, sf * scale_factor)
        center = np.round(
            np.asarray(
                [
                    (bbox[2] / 2.0 + bbox[0]) * sf * scale_factor,
                    (bbox[3] / 2.0 + bbox[1]) * sf * scale_factor,
                ]
            )
        ).astype(int)
        bbox = np.hstack([center - img_size / 2.0, center + img_size / 2.0])
        img = crop(new_img, bbox, bgval=0)
    return img, bbox, scale_factor


def resize_img(img, scale_factor):
    """

    Args:
        img:
        scale_factor:

    Returns:

    """
    new_size = (np.round(np.array(img.shape[:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]),
        new_size[1] / float(img.shape[1]),
    ]
    return new_img, actual_factor


def crop_image(img, img_size=256, bgval=-1):
    """

    Args:
        img:
        img_size:
        bgval:

    Returns:

    """
    # Crop img_size x img_size from the center
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # img center in (x, y)
    center = center[::-1]
    bbox = np.hstack([center - img_size / 2.0, center + img_size / 2.0])
    img = crop(img, bbox, bgval=bgval)
    return img, center, bbox


def peturb_bbox(bbox, pf=0, jf=0):
    """
    Jitters and pads the input bbox.

    Args:
        bbox: Zero-indexed tight bbox.
        pf: padding fraction.
        jf: jittering fraction.
    Returns:
        pet_bbox: Jittered and padded box. Might have -ve or out-of-image coordinates
    """
    pet_bbox = [coord for coord in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    pet_bbox[0] -= (pf * bwidth) + (1 - 2 * np.random.random()) * jf * bwidth
    pet_bbox[1] -= (pf * bheight) + (1 - 2 * np.random.random()) * jf * bheight
    pet_bbox[2] += (pf * bwidth) + (1 - 2 * np.random.random()) * jf * bwidth
    pet_bbox[3] += (pf * bheight) + (1 - 2 * np.random.random()) * jf * bheight

    return pet_bbox


def square_bbox(bbox):
    """
    Converts a bbox to have a square shape by increasing size along non-max dimension.
    """
    sq_bbox = [int(round(coord)) for coord in bbox]
    bwidth = sq_bbox[2] - sq_bbox[0] + 1
    bheight = sq_bbox[3] - sq_bbox[1] + 1
    maxdim = float(max(bwidth, bheight))

    dw_b_2 = int(round((maxdim - bwidth) / 2.0))
    dh_b_2 = int(round((maxdim - bheight) / 2.0))

    sq_bbox[0] -= dw_b_2
    sq_bbox[1] -= dh_b_2
    sq_bbox[2] = sq_bbox[0] + maxdim - 1
    sq_bbox[3] = sq_bbox[1] + maxdim - 1

    return sq_bbox


def crop(img, bbox, bgval=None):
    """
    Crops a region from the image corresponding to the bbox.
    If some regions specified go outside the image boundaries, the pixel values are set to bgval.

    Args:
        img: image to crop
        bbox: bounding box to crop
        bgval: default background for regions outside image; if none add random noise
    """
    bbox = [int(round(c)) for c in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    # TODO: rewrite in an adequate way
    im_shape = np.shape(img)
    im_h, im_w = im_shape[0], im_shape[1]

    nc = 1 if len(im_shape) < 3 else im_shape[2]

    if bgval is None or bgval < 0:
        img_out = np.random.rand(bheight, bwidth, nc)
    else:
        img_out = np.ones((bheight, bwidth, nc)) * bgval

    x_min_src = max(0, bbox[0])
    x_max_src = min(im_w, bbox[2] + 1)
    y_min_src = max(0, bbox[1])
    y_max_src = min(im_h, bbox[3] + 1)

    x_min_trg = x_min_src - bbox[0]
    x_max_trg = x_max_src - x_min_src + x_min_trg
    y_min_trg = y_min_src - bbox[1]
    y_max_trg = y_max_src - y_min_src + y_min_trg

    if len(img.shape) < 3:
        img = img[:, :, np.newaxis]
    img_out[y_min_trg:y_max_trg, x_min_trg:x_max_trg, :] = img[
        y_min_src:y_max_src, x_min_src:x_max_src, :
    ]
    return img_out


def compute_dt(mask):
    """
    Computes distance transform of mask.
    """
    dist = distance_transform_edt(1 - mask) / max(mask.shape)
    return dist


def compute_dt_barrier(mask, k=50):
    """
    Computes barrier distance transform of mask.
    """
    dist_out = distance_transform_edt(1 - mask)
    dist_in = distance_transform_edt(mask)

    dist_diff = (dist_out - dist_in) / max(mask.shape)

    dist = 1.0 / (1 + np.exp(k * -dist_diff))
    return dist
