#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.
"""Trainer for yolox.exp.cb_yolox_exp.CBYOLOXFinetuneExp."""

import os

import torch
from loguru import logger

from .trainer import Trainer

__all__ = ["CBYOLOXFinetuneTrainer"]


def _unwrap(model):
    """Return the module behind DataParallel / DistributedDataParallel."""
    return model.module if hasattr(model, "module") else model


class CBYOLOXFinetuneTrainer(Trainer):
    """
    Trainer for :class:`CBYOLOXFinetuneExp`.

    Re-applies the freeze (requires_grad + frozen BN in eval mode) after
    checkpoint loading and at the start of every epoch while in the frozen
    stage. If ``exp.unfreeze_epoch`` is set, at that epoch the whole model is
    unfrozen and the optimizer is rebuilt with all parameters (fresh SGD state;
    the LR is set by the scheduler every iteration).
    """

    def __init__(self, exp, args):
        super().__init__(exp, args)
        self._unfrozen = False

    def resume_train(self, model):
        # A checkpoint saved after the unfreeze holds a full-model optimizer:
        # switch to that stage before the parent loads the optimizer state.
        if self.args.resume and self.exp.unfreeze_epoch is not None:
            ckpt_file = self.args.ckpt
            if ckpt_file is None:
                ckpt_file = os.path.join(self.file_name, "latest_ckpt.pth")
            if os.path.isfile(ckpt_file):
                ckpt = torch.load(ckpt_file, map_location="cpu")
                start_epoch = (
                    self.args.start_epoch
                    if self.args.start_epoch is not None
                    else ckpt.get("start_epoch", 0)
                )
                del ckpt
                if not self.exp.in_frozen_stage(start_epoch):
                    logger.info(
                        f"Resuming at epoch {start_epoch} (after unfreeze): "
                        "unfreezing the model and rebuilding the optimizer."
                    )
                    self.exp.apply_unfreeze(_unwrap(model))
                    self.optimizer = self.exp.build_optimizer(self.args.batch_size)
                    self._unfrozen = True
        return super().resume_train(model)

    def before_train(self):
        super().before_train()
        if not self._unfrozen:
            self.exp.apply_freeze(_unwrap(self.model), verbose=True)
        self.exp.check_optimizer(self.optimizer)

    def before_epoch(self):
        super().before_epoch()
        model = _unwrap(self.model)
        if self.exp.in_frozen_stage(self.epoch):
            # model.train() re-enables the running stats of frozen BN layers
            self.exp.apply_freeze(model, verbose=False)
        elif not self._unfrozen:
            logger.info(f"Epoch {self.epoch + 1}: unfreezing the whole model.")
            self.exp.apply_unfreeze(model)
            self.optimizer = self.exp.build_optimizer(self.args.batch_size)
            self.exp.check_optimizer(self.optimizer)
            self._unfrozen = True
