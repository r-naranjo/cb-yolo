#!/usr/bin/env python3
"""
CB-YOLOX-L: fine-tune the classification branch + concept bottleneck of a
trained YOLOX-L (everything else frozen).

    python tools/train.py -f exps/cb_yolox/cb_yolox_l.py -d 1 -b 16 --fp16 -c yolox_l.pth

use_concepts_in_simota=True is the "concept SimOTA" variant; for the vanilla
SimOTA variant append `use_concepts_in_simota False` to the command.
"""

import os

from yolox.exp.cb_yolox_exp import CBYOLOXFinetuneExp


class Exp(CBYOLOXFinetuneExp):
    def __init__(self):
        super().__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
