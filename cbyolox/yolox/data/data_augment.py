#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np

from yolox.utils import xyxy2cxcywh


def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets


def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)

    return img, targets


def _mirror(image, boxes, prob=0.5):
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransform:
    """
    Flip + HSV + letterbox for training samples.

    Input targets: (N, 5 + K) rows [x1, y1, x2, y2, class, concepts...].
    Output labels: (max_labels, 5 + K) rows [class, cx, cy, w, h, concepts...].
    Concept columns are carried through unchanged.
    """

    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, num_concepts=None):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.num_concepts = num_concepts if num_concepts is not None else 0

    def __call__(self, image, targets, input_dim):
        has_concepts = targets.shape[1] > 5 if len(targets) > 0 else False

        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        concepts = targets[:, 5:].copy() if has_concepts else None
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5 + self.num_concepts), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        concepts_o = targets_o[:, 5:] if has_concepts else None
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        image_t, boxes = _mirror(image, boxes, self.flip_prob)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]
        concepts_t = concepts[mask_b] if has_concepts else None

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            concepts_t = concepts_o

        labels_t = np.expand_dims(labels_t, 1)

        if has_concepts:
            targets_t = np.hstack((labels_t, boxes_t, concepts_t))
        else:
            targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, targets_t.shape[1]))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels
        num_concepts (int): width of the (dummy) returned labels is 5 + num_concepts

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False, num_concepts=None):
        self.swap = swap
        self.legacy = legacy
        self.num_concepts = num_concepts if num_concepts is not None else 0

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5 + self.num_concepts))


class AlbumentationsTransform:
    """
    Extra augmentations (albumentations 2.0.x, MIT) applied to the raw image before
    TrainTransform. Works on YOLOX targets of shape (N, 5 + K):
    [x1, y1, x2, y2, class, concepts...] in absolute pixel coordinates.

    Applied in order, each with its own probability:
        ElasticTransform, Rotate (+-rotate_limit deg)   (spatial, boxes re-fitted)
        ChannelDropout, GaussianBlur, GaussNoise,
        RandomFog, RandomRain, RandomSnow                (pixel-only)

    Boxes with less than ``min_visibility`` of their area left after the
    spatial transforms are dropped. Horizontal flip and HSV jitter are done by
    TrainTransform as in YOLOX.

    Returns the image and a (max_labels, 5 + K) array in the input format
    (xyxy + class + concepts, zero-padded).
    """

    def __init__(
        self,
        max_labels=50,
        num_concepts=0,
        elastic_p=0.1,
        rotate_p=0.2,
        rotate_limit=10,
        channel_dropout_p=0.05,
        blur_p=0.1,
        gauss_noise_p=0.1,
        fog_p=0.05,
        rain_p=0.05,
        snow_p=0.05,
        min_visibility=0.1,
    ):
        import albumentations as A

        self.max_labels = max_labels
        self.num_concepts = num_concepts if num_concepts is not None else 0

        self.transform = A.Compose(
            [
                A.ElasticTransform(alpha=1.0, sigma=50, p=elastic_p),
                # border_mode=0: constant fill, as YOLOX's affine padding
                A.Rotate(
                    limit=(-rotate_limit, rotate_limit),
                    border_mode=0,
                    p=rotate_p,
                ),
                A.ChannelDropout(p=channel_dropout_p),
                A.GaussianBlur(sigma_limit=(0.5, 1.5), p=blur_p),
                A.GaussNoise(std_range=(0.02, 0.08), p=gauss_noise_p),
                A.RandomFog(fog_coef_range=(0.1, 0.3), p=fog_p),
                A.RandomRain(p=rain_p),
                A.RandomSnow(p=snow_p),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",  # [x1, y1, x2, y2], absolute pixels
                label_fields=["class_labels", "concept_labels"],
                min_visibility=min_visibility,
            ),
        )

    def _empty(self, image):
        augmented = self.transform(image=image, bboxes=[], class_labels=[], concept_labels=[])
        targets_t = np.zeros((self.max_labels, 5 + self.num_concepts), dtype=np.float32)
        return augmented["image"], targets_t

    def __call__(self, image, targets, input_dim):
        K = self.num_concepts
        n_cols = 5 + K

        if len(targets) == 0:
            return self._empty(image)

        bboxes = targets[:, :4].copy()
        h, w = image.shape[:2]
        # Albumentations rejects coordinates outside the image
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, w)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, h)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, w)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, h)
        valid = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
        bboxes = bboxes[valid].tolist()
        class_labels = targets[valid, 4].tolist()
        # concept vectors travel through Albumentations as strings
        if K > 0:
            concept_labels = [",".join(map(str, row)) for row in targets[valid, 5:5 + K]]
        else:
            concept_labels = ["" for _ in range(len(bboxes))]

        if len(bboxes) == 0:
            return self._empty(image)

        augmented = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels,
            concept_labels=concept_labels,
        )
        image_t = augmented["image"]
        aug_bboxes = augmented["bboxes"]
        aug_classes = augmented["class_labels"]
        aug_concepts = augmented["concept_labels"]

        num_objs = len(aug_bboxes)
        targets_t = np.zeros((self.max_labels, n_cols), dtype=np.float32)
        if num_objs == 0:
            return image_t, targets_t

        targets_arr = np.zeros((num_objs, n_cols), dtype=np.float32)
        for i in range(num_objs):
            targets_arr[i, 0:4] = aug_bboxes[i]
            targets_arr[i, 4] = aug_classes[i]
            if K > 0 and aug_concepts[i]:
                cv = np.array(aug_concepts[i].split(","), dtype=np.float32)
                targets_arr[i, 5:5 + len(cv)] = cv[:K]

        n = min(num_objs, self.max_labels)
        targets_t[:n] = targets_arr[:n]
        return image_t, targets_t


class ComposedTrainTransform:
    """Apply AlbumentationsTransform, then TrainTransform."""

    def __init__(self, albu_transform, train_transform):
        self.albu_transform = albu_transform
        self.train_transform = train_transform

    def __call__(self, image, targets, input_dim):
        image, targets = self.albu_transform(image, targets, input_dim)
        return self.train_transform(image, targets, input_dim)
