#!/usr/bin/env python3
"""
CB-YOLOX-S trained end-to-end (all layers, standard YOLOX schedule and
augmentations + Albumentations), without freezing. Starts from random weights
unless `-c <ckpt>` or `pretrained_ckpt` is given.

    python tools/train.py -f exps/cb_yolox/cb_yolox_s_scratch.py -d 8 -b 64 --fp16
"""

import os

from yolox.exp.cb_yolox_exp import CBYOLOXExp


class Exp(CBYOLOXExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.50
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (14, 26)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
