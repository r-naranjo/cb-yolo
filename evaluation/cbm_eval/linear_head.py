"""Closed-form re-evaluation of the concept -> class layer.

In the CB-YOLOX head the classification logits of one anchor are

    concept_probs = sigmoid(concept_logits)               # (K,)
    cls_logits    = concept_to_cls[s](concept_probs)       # 1x1 conv, K -> C

where s is the FPN level of the anchor. A 1x1 convolution only mixes the K
channels at the same spatial location, so for one anchor
`cls_logits = W_s @ probs + b_s` exactly. Editing concept probabilities and
re-applying (W_s, b_s) is therefore identical to re-running the network with
the same edit applied through `head.set_interventions()` (which overwrites
the raw concept logits right before this conv), without another forward
pass. The faithfulness analyses use this to evaluate whole
deactivation/intervention curves from a single inference pass.

The FPN level of every kept detection is `scale_info[0]["scale_map"]`, in the
same order as `model.head.concept_to_cls`.
"""

import numpy as np


def extract_concept_to_cls_weights(model):
    """Per-level (W (C, K), b (C,)) float32 arrays of head.concept_to_cls."""
    weights = []
    for conv in model.head.concept_to_cls:
        W = conv.weight.detach().cpu().float().numpy().reshape(conv.out_channels, conv.in_channels)
        b = conv.bias.detach().cpu().float().numpy()
        weights.append((W, b))
    return weights


def cls_logits_from_concept_probs(concept_probs, scale_idx, scale_weights):
    """Vectorised `W_s @ probs + b_s`. concept_probs: (N, K); scale_idx: (N,).
    Returns (N, C) float32."""
    num_classes = scale_weights[0][0].shape[0]
    n = concept_probs.shape[0]
    cls_logits = np.empty((n, num_classes), dtype=np.float32)
    for s, (W, b) in enumerate(scale_weights):
        mask = scale_idx == s
        if not np.any(mask):
            continue
        cls_logits[mask] = concept_probs[mask].astype(np.float32) @ W.T + b
    return cls_logits


def topk_hits(cls_logits, gt_classes, ks=(1, 3, 5)):
    """For each k in `ks`, a boolean (N,) array: is gt_classes among the k
    highest cls_logits of the row."""
    max_k = max(ks)
    top_idx = np.argpartition(-cls_logits, max_k - 1, axis=1)[:, :max_k]
    row_order = np.argsort(-np.take_along_axis(cls_logits, top_idx, axis=1), axis=1)
    top_idx = np.take_along_axis(top_idx, row_order, axis=1)
    gt_col = gt_classes[:, None]
    return {k: (top_idx[:, :k] == gt_col).any(axis=1) for k in ks}
