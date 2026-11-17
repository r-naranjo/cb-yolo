#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.
"""
Base experiments for CB-YOLOX (YOLOX with a concept bottleneck in the
classification branch).

- :class:`CBYOLOXExp`: full CB-YOLOX training. Adds the concept bottleneck
  configuration, the Albumentations augmentations used in all CB-YOLOX runs,
  the L1 loss from the first epoch, and optional loading of a pretrained
  checkpoint (``pretrained_ckpt``).
- :class:`CBYOLOXFinetuneExp`: fine-tunes only the classification branch
  (``cls_convs`` and the concept bottleneck ``concept_preds`` /
  ``concept_to_cls``) of a trained YOLOX model. Everything else (backbone, FPN,
  ``stems``, regression and objectness branches) is frozen and the frozen
  BatchNorm layers are kept in eval mode. Optionally the whole model is
  unfrozen at epoch ``unfreeze_epoch``. Uses
  :class:`yolox.core.cb_finetune_trainer.CBYOLOXFinetuneTrainer`.

The experiment files in ``exps/cb_yolox/`` derive from these classes.
"""

import os

import torch
import torch.nn as nn
from loguru import logger

from .yolox_base import Exp

__all__ = ["CBYOLOXExp", "CBYOLOXFinetuneExp"]


class CBYOLOXExp(Exp):
    def __init__(self):
        super().__init__()

        # ---------------- concept bottleneck ---------------- #
        # 143 concepts for the VLM (Qwen3-VL) annotations; 400 for the
        # CLIP per-class descriptor annotations (80 classes x 5 descriptors).
        self.num_concepts = 143
        self.concept_loss_weight = 1.0
        self.concept_loss_type = "asl"
        # SimOTA with the concept cost ("concept SimOTA"). Set to False for the
        # vanilla SimOTA cost (cls + 3 * iou), e.g. with `-o` on the command line
        # (tools/train.py ... use_concepts_in_simota False).
        self.use_concepts_in_simota = True
        self.simota_concept_weight = 0.6148

        # The L1 box loss is active from the first epoch (YOLOX only enables it
        # for the last no-aug epochs).
        self.l1_loss_from_start = True

        # Optional checkpoint (e.g. an official YOLOX model) loaded in
        # get_model() with yolox.utils.load_ckpt (non-matching layers are
        # skipped). Equivalent to passing `-c <ckpt>` to tools/train.py.
        self.pretrained_ckpt = None

        # ---------------- dataloader / logging ---------------- #
        self.data_num_workers = 8
        self.print_interval = 5

    # ------------------------------------------------------------------ #
    def merge(self, cfg_list):
        # BaseExp.merge casts with bool("False") == True; parse booleans here.
        cfg_list = list(cfg_list)
        for i in range(0, len(cfg_list) - 1, 2):
            k, v = cfg_list[i], cfg_list[i + 1]
            if isinstance(getattr(self, k, None), bool) and isinstance(v, str):
                if v.lower() not in ("true", "false", "1", "0"):
                    raise ValueError(f"{k}: expected a boolean, got {v!r}")
                cfg_list[i + 1] = v.lower() in ("true", "1")
            elif k in ("pretrained_ckpt",) and isinstance(v, str) and v.lower() == "none":
                cfg_list[i + 1] = None
        super().merge(cfg_list)

    def get_model(self):
        model = super().get_model()
        if self.l1_loss_from_start:
            model.head.use_l1 = True
        self.load_pretrained(model)
        return model

    def load_pretrained(self, model):
        """Load ``self.pretrained_ckpt`` (if set) into ``model``."""
        from yolox.utils import load_ckpt

        if not self.pretrained_ckpt:
            return model
        if not os.path.isfile(self.pretrained_ckpt):
            raise FileNotFoundError(f"pretrained_ckpt not found: {self.pretrained_ckpt}")
        ckpt = torch.load(self.pretrained_ckpt, map_location="cpu")
        load_ckpt(model, ckpt.get("model", ckpt))
        logger.info(f"Loaded pretrained weights from {self.pretrained_ckpt}")
        return model

    def get_train_transform(self, no_aug=False):
        """AlbumentationsTransform followed by TrainTransform (flip + HSV)."""
        from yolox.data import TrainTransform
        from yolox.data.data_augment import AlbumentationsTransform, ComposedTrainTransform

        if no_aug:
            # all extra augmentations disabled (boxes are still clipped/filtered)
            albu_transform = AlbumentationsTransform(
                max_labels=120,
                num_concepts=self.num_concepts,
                elastic_p=0.0,
                rotate_p=0.0,
                channel_dropout_p=0.0,
                blur_p=0.0,
                gauss_noise_p=0.0,
                fog_p=0.0,
                rain_p=0.0,
                snow_p=0.0,
            )
        else:
            albu_transform = AlbumentationsTransform(
                max_labels=120,
                num_concepts=self.num_concepts,
                rotate_p=0.5,
                rotate_limit=5,
                elastic_p=0.5,
                channel_dropout_p=0.5,
                blur_p=0,
                gauss_noise_p=0,
                fog_p=0,
                rain_p=0,
                snow_p=0,
            )
        train_transform = TrainTransform(
            max_labels=120,
            flip_prob=self.flip_prob,
            hsv_prob=self.hsv_prob,
            num_concepts=self.num_concepts,
        )
        return ComposedTrainTransform(albu_transform, train_transform)

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import COCODataset

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            preproc=self.get_train_transform(),
            cache=cache,
            cache_type=cache_type,
        )


class CBYOLOXFinetuneExp(CBYOLOXExp):
    """
    Fine-tune only the classification branch (+ concept bottleneck) of a
    trained YOLOX model. A full YOLOX checkpoint is required, either via
    ``pretrained_ckpt`` or ``-c`` in tools/train.py, because the regression
    and objectness branches never train while frozen.
    """

    def __init__(self):
        super().__init__()

        # ---------------- augmentation ---------------- #
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.enable_mixup = False
        self.shear = 0
        self.degrees = 0
        self.translate = 0
        self.hsv_prob = 0.8
        self.flip_prob = 0.5
        self.no_aug_epochs = 5

        # multi-scale training: 448 .. 832
        self.random_size = (14, 26)

        # ---------------- what is trained ---------------- #
        # Parameter-name prefixes that stay trainable; everything else is frozen.
        # `head.stems` is shared by the cls and reg branches and stays frozen.
        self.trainable_param_prefixes = [
            "head.cls_convs",
            "head.cls_preds",  # plain head only (num_concepts=None)
        ]
        # also train g (concept_preds) and f (concept_to_cls)
        self.train_concept_branch = True
        # keep BatchNorm layers whose parameters are frozen in eval mode
        self.freeze_bn_running_stats = True
        # None: frozen for the whole run. An int N: the whole model is
        # unfrozen (and the optimizer rebuilt) at the start of epoch N.
        self.unfreeze_epoch = None

    # ------------------------------------------------------------------ #
    # freezing
    # ------------------------------------------------------------------ #
    def in_frozen_stage(self, epoch):
        return self.unfreeze_epoch is None or epoch < self.unfreeze_epoch

    def trainable_prefixes(self):
        prefixes = list(self.trainable_param_prefixes)
        if self.train_concept_branch and self.num_concepts is not None and self.use_cbm:
            prefixes += ["head.concept_preds", "head.concept_to_cls"]
        return tuple(prefixes)

    def apply_freeze(self, model, verbose=True):
        """
        requires_grad=False on all parameters except the trainable prefixes;
        BN layers with frozen parameters in eval mode. Idempotent.
        """
        prefixes = self.trainable_prefixes()
        n_train, n_frozen = 0, 0
        for name, p in model.named_parameters():
            if name.startswith(prefixes):
                p.requires_grad = True
                n_train += p.numel()
            else:
                p.requires_grad = False
                n_frozen += p.numel()

        n_bn_eval = 0
        if self.freeze_bn_running_stats:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                    if not any(p.requires_grad for p in m.parameters(recurse=False)):
                        m.eval()
                        n_bn_eval += 1

        if n_train == 0:
            raise ValueError(
                f"No trainable parameters left; check trainable_param_prefixes {prefixes}."
            )
        if verbose:
            total = n_train + n_frozen
            logger.info(
                f"Trainable params: {n_train:,} / {total:,} ({n_train / max(total, 1):.2%}), "
                f"frozen BN layers in eval mode: {n_bn_eval}, prefixes: {prefixes}"
            )
        return n_train, n_frozen

    def apply_unfreeze(self, model):
        """Everything trainable, every BN back in train mode."""
        for p in model.parameters():
            p.requires_grad = True
        model.train()

    def check_optimizer(self, optimizer):
        """No frozen parameter may be in the optimizer (weight decay would still change it)."""
        frozen = [p for g in optimizer.param_groups for p in g["params"] if not p.requires_grad]
        if frozen:
            raise RuntimeError(f"{len(frozen)} frozen tensors are inside the optimizer.")

    # ------------------------------------------------------------------ #
    # model / optimizer / trainer
    # ------------------------------------------------------------------ #
    def get_model(self):
        model = super().get_model()
        if not self.pretrained_ckpt:
            logger.warning(
                "pretrained_ckpt is None: pass a trained YOLOX checkpoint with "
                "`-c <ckpt>` to tools/train.py (the frozen box/objectness branches "
                "are never trained)."
            )
        self.apply_freeze(model, verbose=True)
        model.train()
        # train() re-enables every BN; put the frozen ones back in eval mode
        self.apply_freeze(model, verbose=False)
        return model

    def build_optimizer(self, batch_size):
        """
        SGD over the parameters that currently require grad, with YOLOX's
        BN-weight / weight (decay) / bias groups. Empty groups are skipped.
        """
        lr = self.warmup_lr if self.warmup_epochs > 0 else self.basic_lr_per_img * batch_size

        pg0, pg1, pg2 = [], [], []  # BN weights | conv weights | biases
        for k, v in self.model.named_modules():
            if not any(p.requires_grad for p in v.parameters(recurse=False)):
                continue
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter) and v.bias.requires_grad:
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                if (hasattr(v, "weight") and isinstance(v.weight, nn.Parameter)
                        and v.weight.requires_grad):
                    pg0.append(v.weight)  # no decay
            elif (hasattr(v, "weight") and isinstance(v.weight, nn.Parameter)
                    and v.weight.requires_grad):
                pg1.append(v.weight)  # apply decay

        groups = [
            (group, extra)
            for group, extra in ((pg0, {}), (pg1, {"weight_decay": self.weight_decay}), (pg2, {}))
            if group
        ]
        if not groups:
            raise ValueError("No trainable parameter reached the optimizer.")

        (params, extra), rest = groups[0], groups[1:]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=self.momentum, nesterov=True, **extra)
        for group, extra in rest:
            optimizer.add_param_group({"params": group, **extra})

        self.optimizer = optimizer
        return optimizer

    def get_optimizer(self, batch_size):
        if getattr(self, "optimizer", None) is None:
            self.build_optimizer(batch_size)
        return self.optimizer

    def get_trainer(self, args):
        # imported here so that loading the exp does not require the training deps
        from yolox.core.cb_finetune_trainer import CBYOLOXFinetuneTrainer

        return CBYOLOXFinetuneTrainer(self, args)
