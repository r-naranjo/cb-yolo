#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Concept Bottleneck extensions for CB-YOLOX.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, cxcywh2xyxy, meshgrid, visualize_assign

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv

# timm provides the Asymmetric Loss (Ridnik et al., ICCV 2021) used for the
# concept branch. It is only required when num_concepts is set and
# concept_loss_type == "asl".
try:
    from timm.loss import AsymmetricLossMultiLabel
    _TIMM_ASL_IMPORT_ERROR = None
except ImportError as _e:  # pragma: no cover
    AsymmetricLossMultiLabel = None
    _TIMM_ASL_IMPORT_ERROR = _e


class YOLOXHead(nn.Module):
    """
    YOLOX decoupled head with an optional sequential Concept Bottleneck (CBM)
    in the classification branch.

    With ``num_concepts=None`` the head is the vanilla YOLOX head. With
    ``num_concepts=K`` the classification branch becomes, per FPN level::

        cls_feat = cls_convs[k](stems[k](x))
        c        = concept_preds[k](cls_feat)          # g: K concept logits
        y        = concept_to_cls[k](sigmoid(c))        # f: 1x1 conv -> class logits

    The plain ``cls_preds`` layer (features -> 80 classes) exists only in the
    vanilla head; the CBM head replaces it with g and f. ``cls_preds`` entries
    in a checkpoint (from YOLOX weights or older CB-YOLOX checkpoints) are
    ignored when loading into a CBM head.

    Training loss (CBM)::

        L = 5 * L_iou + L_obj + L_cls + L_l1 + concept_loss_weight * L_concept

    where L_concept is the ASL (default) or BCE loss between the concept
    logits of the foreground anchors and the concept vector of their
    assigned ground truth, normalised by the number of foreground anchors.

    SimOTA assignment cost::

        c_ij = L_cls_ij + simota_iou_weight * L_iou_ij
               [+ simota_concept_weight * L_concept_ij]

    The concept term (BCE summed over the K concepts) is only added when
    ``use_concepts_in_simota`` is True.

    Test-time interventions: see :meth:`set_interventions`.
    """

    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        num_concepts=None,
        concept_loss_weight=1.269,
        use_concepts_in_simota=True,
        concept_loss_type="asl",
        asl_gamma_neg=4.0,
        asl_gamma_pos=1.0,
        asl_clip=0.05,
        asl_eps=1e-8,
        simota_iou_weight=3.0,
        simota_concept_weight=0.6148,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
            num_concepts (int | None): number of concepts K. None builds the
                vanilla YOLOX head.
            concept_loss_weight (float): weight w_c of the concept loss.
            use_concepts_in_simota (bool): add the concept cost to the SimOTA
                assignment cost. Does not affect the concept loss itself.
            concept_loss_type (str): "asl" (timm AsymmetricLossMultiLabel) or
                "bce" (binary cross-entropy with logits).
            asl_gamma_neg (float): ASL focusing parameter for negative concepts.
            asl_gamma_pos (float): ASL focusing parameter for positive concepts.
            asl_clip (float): ASL probability margin; negatives with p < clip
                contribute zero loss.
            asl_eps (float): numerical floor inside the ASL log.
            simota_iou_weight (float): weight (lambda) of the IoU cost in SimOTA.
            simota_concept_weight (float): weight (sigma) of the concept cost in
                SimOTA (only used when use_concepts_in_simota is True).
        """
        super().__init__()

        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        # ---------------- concept bottleneck config ---------------- #
        self.num_concepts = num_concepts
        self.concept_loss_weight = concept_loss_weight
        # Value of the last computed concept loss (for logging by the trainer).
        self.last_concept_loss = 0.0
        # Raw concept logits per FPN level, set on every forward (None without CBM).
        self.concept_outputs = None
        # Test-time interventions, see set_interventions().
        self.interventions = None
        self.interventions_grad = False

        self.use_concepts_in_simota = use_concepts_in_simota
        self.simota_iou_weight = simota_iou_weight
        self.simota_concept_weight = simota_concept_weight

        self.concept_loss_type = str(concept_loss_type).lower()
        if self.concept_loss_type not in ("asl", "bce"):
            raise ValueError(
                f"concept_loss_type must be 'asl' or 'bce', got {concept_loss_type!r}"
            )
        self.asl_gamma_neg = asl_gamma_neg
        self.asl_gamma_pos = asl_gamma_pos
        self.asl_clip = asl_clip
        self.asl_eps = asl_eps
        self.concept_loss_fn = None

        if self.num_concepts is not None and self.concept_loss_type == "asl":
            if AsymmetricLossMultiLabel is None:
                raise ImportError(
                    "concept_loss_type='asl' requires timm (`pip install timm`), "
                    f"but the import failed: {_TIMM_ASL_IMPORT_ERROR}"
                )
            # Stateless module (no parameters or buffers).
            self.concept_loss_fn = AsymmetricLossMultiLabel(
                gamma_neg=self.asl_gamma_neg,
                gamma_pos=self.asl_gamma_pos,
                clip=self.asl_clip,
                eps=self.asl_eps,
                disable_torch_grad_focal_loss=False,
            )

        if self.num_concepts is not None:
            self.concept_preds = nn.ModuleList()   # g: x -> c
            self.concept_to_cls = nn.ModuleList()  # f: c -> y

        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            if self.num_concepts is None:
                self.cls_preds.append(
                    nn.Conv2d(
                        in_channels=int(256 * width),
                        out_channels=self.num_classes,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            else:
                # g: concept logits
                self.concept_preds.append(
                    nn.Conv2d(
                        in_channels=int(256 * width),
                        out_channels=self.num_concepts,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
                # f: concept probabilities -> class logits
                self.concept_to_cls.append(
                    nn.Conv2d(
                        in_channels=self.num_concepts,
                        out_channels=self.num_classes,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def set_interventions(self, interventions):
        """
        Set test-time concept interventions applied in every forward pass.

        Args:
            interventions (list): ``[[anchor_idx, [concept_ids], [values]], ...]``.
                ``anchor_idx`` is the flat anchor index over all FPN levels
                (level order stride 8, 16, 32; row-major inside a level, i.e.
                0..8399 for a 640x640 input). The listed concept logits of that
                anchor are overwritten with ``values`` (raw logits, applied
                before the sigmoid) for every image in the batch. ``values``
                may be a scalar or a length-1 list, which is broadcast to all
                concepts. Invalid entries are skipped with a warning.

        If ``self.interventions_grad`` is False (default) the overwrite is done
        under ``torch.no_grad()``.
        """
        self.interventions = interventions

    def clear_interventions(self):
        self.interventions = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # The CBM head has no plain cls_preds: drop it from checkpoints that
        # still contain it, so they load strictly.
        if self.num_concepts is not None:
            for key in [k for k in state_dict if k.startswith(prefix + "cls_preds.")]:
                del state_dict[key]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _apply_interventions(self, concept_output, anchor_base):
        """Overwrite concept logits in-place for the anchors of this FPN level."""
        _, KC, H, W = concept_output.shape
        n_anchors_this_scale = H * W

        ctx = torch.enable_grad() if self.interventions_grad else torch.no_grad()
        with ctx:
            for intervention in self.interventions:
                if not (isinstance(intervention, (list, tuple)) and len(intervention) >= 3):
                    logger.warning(f"Invalid intervention format (skipped): {intervention}")
                    continue

                try:
                    anchor_idx = int(intervention[0])
                except Exception:
                    logger.warning(f"Invalid anchor_idx in intervention (skipped): {intervention}")
                    continue

                concepts = intervention[1]
                values = intervention[2]

                if isinstance(concepts, torch.Tensor):
                    concepts = concepts.tolist()
                if isinstance(values, torch.Tensor):
                    values = values.tolist()

                # broadcast a single value to all concepts
                if not hasattr(values, "__len__") or isinstance(values, (int, float)):
                    values = [float(values)] * (len(concepts))
                elif len(values) != len(concepts):
                    if len(values) == 1:
                        values = [float(values[0])] * len(concepts)
                    else:
                        logger.warning(f"Length mismatch concepts/values (skipped): {intervention}")
                        continue

                # only anchors that belong to this level
                if not (anchor_base <= anchor_idx < anchor_base + n_anchors_this_scale):
                    continue

                local_idx = anchor_idx - anchor_base
                h_idx = int(local_idx // W)
                w_idx = int(local_idx % W)

                if not (0 <= h_idx < H and 0 <= w_idx < W):
                    logger.warning(
                        f"Computed (h,w) out of range for intervention (skipped): {intervention}"
                    )
                    continue

                for concept_idx_raw, concept_val in zip(concepts, values):
                    try:
                        concept_idx = int(concept_idx_raw)
                    except Exception:
                        logger.warning(f"Invalid concept_idx in intervention (skipped): {intervention}")
                        continue

                    if not (0 <= concept_idx < KC):
                        logger.warning(
                            f"Concept index {concept_idx} out of range (0..{KC - 1}), "
                            "intervention skipped."
                        )
                        continue

                    # applied to every image of the batch
                    concept_output[:, concept_idx, h_idx, w_idx] = float(concept_val)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        concept_outputs = []
        # flat index of the first anchor of the current FPN level
        anchor_base = 0

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)

            if self.num_concepts is not None:
                # g: concept logits (B, K, H, W)
                concept_output = self.concept_preds[k](cls_feat)
                if self.interventions:
                    self._apply_interventions(concept_output, anchor_base)
                # f: sigmoid(concepts) -> class logits (B, C, H, W)
                cls_output = self.concept_to_cls[k](torch.sigmoid(concept_output))
                concept_outputs.append(concept_output)
            else:
                cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, 1, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)
            anchor_base += cls_output.shape[-2] * cls_output.shape[-1]

        self.concept_outputs = concept_outputs if self.num_concepts is not None else None

        if self.training:
            concept_preds_cat = None
            if self.num_concepts is not None:
                # (B, K, H, W) per level -> (B, n_anchors_all, K)
                concept_preds_cat = torch.cat(
                    [
                        c.view(c.shape[0], c.shape[1], -1).permute(0, 2, 1)
                        for c in concept_outputs
                    ],
                    dim=1,
                )
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
                concept_preds=concept_preds_cat,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(outputs[..., 2:4]) * strides,
            outputs[..., 4:]
        ], dim=-1)
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
        concept_preds=None,
    ):
        """
        labels: (B, max_labels, 5 + K) rows [cls, cx, cy, w, h, concepts...].
        concept_preds: (B, n_anchors_all, K) concept logits or None.
        """
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        concept_targets = []
        has_concepts = concept_preds is not None and self.num_concepts > 0
        # The concept loss is always computed with concepts; the SimOTA cost
        # only uses them when use_concepts_in_simota is set.
        simota_concepts = has_concepts and self.use_concepts_in_simota

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                if simota_concepts:
                    concept_kwargs = dict(
                        concept_preds=concept_preds[batch_idx],
                        gt_concepts=labels[batch_idx, :num_gt, 5:5 + self.num_concepts],
                    )
                else:
                    concept_kwargs = {}

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                        **concept_kwargs,
                    )
                except RuntimeError as e:
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise  # RuntimeError might not caused by CUDA OOM

                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                        "cpu",
                        **concept_kwargs,
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if has_concepts:
                    # concept target of each foreground anchor: (num_fg_img, K)
                    gt_concepts_per_image = labels[batch_idx, :num_gt, 5:5 + self.num_concepts]
                    concept_targets.append(gt_concepts_per_image[matched_gt_inds].to(dtype))
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        loss_concept = 0.0
        if has_concepts:
            if concept_preds.size(2) != self.num_concepts:
                raise ValueError(
                    f"concept_preds has {concept_preds.size(2)} concepts, "
                    f"expected {self.num_concepts}"
                )
            concept_preds_flat = concept_preds.view(-1, self.num_concepts)[fg_masks]

            if concept_targets:
                concept_targets_cat = torch.cat(concept_targets, 0).to(concept_preds_flat.dtype)
                if concept_preds_flat.shape != concept_targets_cat.shape:
                    raise ValueError(
                        f"Shape mismatch: concept_preds {tuple(concept_preds_flat.shape)}, "
                        f"concept_targets {tuple(concept_targets_cat.shape)}"
                    )
                # Both losses are a SUM over foreground anchors and concepts,
                # normalised by num_fg (timm's ASL returns -loss.sum()).
                if self.concept_loss_type == "asl":
                    # ASL is computed in fp32: it takes log(sigmoid(x))
                    # explicitly and the sum can overflow fp16.
                    with torch.cuda.amp.autocast(enabled=False):
                        concept_loss_raw = self.concept_loss_fn(
                            concept_preds_flat.float(),
                            concept_targets_cat.float(),
                        )
                else:
                    concept_loss_raw = F.binary_cross_entropy_with_logits(
                        concept_preds_flat,
                        concept_targets_cat,
                        reduction="sum",
                    )
                loss_concept = concept_loss_raw / num_fg
            else:
                loss_concept = torch.tensor(0.0, device=outputs.device, dtype=outputs.dtype)

        reg_weight = 5.0
        if has_concepts:
            loss = (
                loss_obj + loss_cls + loss_l1
                + self.concept_loss_weight * loss_concept
                + reg_weight * loss_iou
            )
        else:
            loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        if isinstance(loss_concept, torch.Tensor):
            self.last_concept_loss = loss_concept.detach().item()
        else:
            self.last_concept_loss = loss_concept

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        obj_preds,
        mode="gpu",
        concept_preds=None,
        gt_concepts=None,
    ):
        """
        SimOTA assignment for one image.

        concept_preds: (n_anchors_all, K) concept logits of this image, or None.
        gt_concepts: (num_gt, K) ground-truth concept vectors, or None.
        When both are given, the concept cost is added to the assignment cost.
        """
        if mode == "cpu":
            print("-----------Using CPU for the Current Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()
            if concept_preds is not None:
                concept_preds = concept_preds.cpu().float()
            if gt_concepts is not None:
                gt_concepts = gt_concepts.cpu().float()

        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_

        # Concept cost: BCE between each candidate anchor's concept logits and
        # each GT concept vector, summed over K -> (num_gt, num_in_boxes_anchor).
        # It stays BCE even when the concept loss is ASL, because timm's ASL
        # only returns a scalar.
        pair_wise_concept_loss = 0
        if concept_preds is not None and gt_concepts is not None:
            with torch.cuda.amp.autocast(enabled=False):
                concept_preds_ = concept_preds[fg_mask].float()
                concept_preds_exp = concept_preds_.unsqueeze(0).expand(num_gt, -1, -1)
                gt_concepts_exp = gt_concepts.float().unsqueeze(1).expand(-1, num_in_boxes_anchor, -1)
                pair_wise_concept_loss = F.binary_cross_entropy_with_logits(
                    concept_preds_exp,
                    gt_concepts_exp,
                    reduction="none"
                ).sum(-1)

        # c_ij = L_cls + lambda * L_iou (+ sigma * L_concept) + geometry constraint
        cost = (
            pair_wise_cls_loss
            + self.simota_iou_weight * pair_wise_ious_loss
            + self.simota_concept_weight * pair_wise_concept_loss
            + float(1e6) * (~geometry_relation)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_geometry_constraint(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
    ):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        # in fixed center
        center_radius = 1.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def visualize_assign_result(self, xin, labels=None, imgs=None, save_prefix="assign_vis_"):
        # original forward logic
        outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []
        # TODO: use forward logic here.

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            if self.num_concepts is not None:
                cls_output = self.concept_to_cls[k](torch.sigmoid(self.concept_preds[k](cls_feat)))
            else:
                cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.full((1, grid.shape[1]), stride_this_level).type_as(xin[0])
            )
            outputs.append(output)

        outputs = torch.cat(outputs, 1)
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]   # [batch, n_anchors_all, n_cls]

        # calculate targets
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        for batch_idx, (img, num_gt, label) in enumerate(zip(imgs, nlabel, labels)):
            img = imgs[batch_idx].permute(1, 2, 0).to(torch.uint8)
            num_gt = int(num_gt)
            if num_gt == 0:
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = label[:num_gt, 1:5]
                gt_classes = label[:num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                _, fg_mask, _, matched_gt_inds, _ = self.get_assignments(  # noqa
                    batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                    bboxes_preds_per_image, expanded_strides, x_shifts,
                    y_shifts, cls_preds, obj_preds,
                )

            img = img.cpu().numpy().copy()  # copy is crucial here
            coords = torch.stack([
                ((x_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
                ((y_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
            ], 1)

            xyxy_boxes = cxcywh2xyxy(gt_bboxes_per_image)
            save_name = save_prefix + str(batch_idx) + ".png"
            img = visualize_assign(img, xyxy_boxes, coords, matched_gt_inds, save_name)
            logger.info(f"save img to {save_name}")
