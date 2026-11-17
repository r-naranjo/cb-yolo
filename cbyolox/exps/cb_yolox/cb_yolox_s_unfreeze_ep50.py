#!/usr/bin/env python3
"""
CB-YOLOX-S, two stages in one run: epochs 0-49 train only the classification
branch + concept bottleneck of a trained YOLOX-S (as cb_yolox_s.py); from
epoch 50 the whole model is trained (the optimizer is rebuilt with all
parameters). Uses the vanilla SimOTA cost.

    python tools/train.py -f exps/cb_yolox/cb_yolox_s_unfreeze_ep50.py -d 1 -b 16 --fp16 -c yolox_s.pth
"""

import os

from yolox.exp.cb_yolox_exp import CBYOLOXFinetuneExp


class Exp(CBYOLOXFinetuneExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.50
        self.use_concepts_in_simota = False
        self.unfreeze_epoch = 50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
