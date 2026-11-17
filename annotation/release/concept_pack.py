"""Compact storage of COCO concept annotations (the released data format).

A concept pack is a ``.npz`` file that holds only the concept labels; boxes,
images and categories come from the official COCO ``instances_*.json``. It
contains:

``annotation_ids``  int64 (N,)  every annotation id, in the order of the
                                annotated JSON
``labelled``        bool  (N,)  whether that annotation has a concept vector
``concepts``        uint8       concept vectors of the labelled annotations, in
                                the same order: bit-packed rows (``packbits``,
                                M x ceil(K/8)) when ``encoding == "packbits"``
                                (binary labels), else dense (M, K)
``encoding``        str         ``"packbits"`` or ``"dense"``
``num_concepts``    int         K
``meta``            str         JSON: format version, source/vocabulary
                                descriptions, extra top-level keys of the
                                annotated JSON (e.g. ``concept_names``)
"""

import json

import numpy as np

FORMAT_VERSION = 1
STANDARD_KEYS = ("info", "licenses", "images", "annotations", "categories")
# image fields read by the training / evaluation code
IMAGE_KEYS_USED = ("id", "file_name", "width", "height")


def pack(annotated, description="", vocabulary=""):
    """Build the arrays of a concept pack from an annotated COCO dict."""
    anns = annotated["annotations"]
    ids = np.array([a["id"] for a in anns], dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("annotation ids are not unique")
    labelled = np.array(["concept_vector" in a for a in anns], dtype=bool)
    vectors = [a["concept_vector"] for a in anns if "concept_vector" in a]
    if not vectors:
        raise ValueError("no annotation has a concept_vector")
    lengths = {len(v) for v in vectors}
    if len(lengths) != 1:
        raise ValueError(f"concept vectors have different lengths: {sorted(lengths)}")
    k = lengths.pop()

    dense = np.asarray(vectors)
    if not np.all(dense == np.round(dense)) or dense.min() < 0 or dense.max() > 255:
        raise ValueError("concept values must be integers in [0, 255]")
    dense = dense.astype(np.uint8)
    if dense.max() <= 1:
        encoding, concepts = "packbits", np.packbits(dense, axis=1)
    else:
        encoding, concepts = "dense", dense

    meta = {
        "format_version": FORMAT_VERSION,
        "description": description,
        "vocabulary": vocabulary,
        "extra_keys": {key: v for key, v in annotated.items() if key not in STANDARD_KEYS},
    }
    return {
        "annotation_ids": ids,
        "labelled": labelled,
        "concepts": concepts,
        "encoding": np.array(encoding),
        "num_concepts": np.array(k),
        "meta": np.array(json.dumps(meta)),
    }


def save_pack(path, arrays):
    np.savez_compressed(path, **arrays)


def load_pack(path):
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    arrays["encoding"] = str(arrays["encoding"])
    arrays["num_concepts"] = int(arrays["num_concepts"])
    arrays["meta"] = json.loads(str(arrays["meta"]))
    if arrays["meta"].get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported concept pack version: {arrays['meta'].get('format_version')}")
    return arrays


def unpack_concepts(arrays):
    """Return the (M, K) uint8 matrix of concept vectors."""
    k = arrays["num_concepts"]
    if arrays["encoding"] == "packbits":
        return np.unpackbits(arrays["concepts"], axis=1, count=k)
    return arrays["concepts"]


def merge(coco, arrays):
    """Return ``coco`` (official instances dict) with the concept labels added.

    Annotations are put in the stored order; each labelled one gets a
    ``concept_vector`` list of ints. Extra top-level keys of the original
    annotated JSON are restored.
    """
    by_id = {a["id"]: a for a in coco["annotations"]}
    ids = arrays["annotation_ids"].tolist()
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} annotation ids are not in the COCO file "
                         f"(e.g. {missing[:5]}); is it the right split?")
    if len(ids) != len(by_id):
        raise ValueError(f"the pack covers {len(ids)} annotations, the COCO file has {len(by_id)}")

    vectors = iter(unpack_concepts(arrays).tolist())
    annotations = []
    for ann_id, has_vector in zip(ids, arrays["labelled"].tolist()):
        ann = dict(by_id[ann_id])
        if has_vector:
            ann["concept_vector"] = next(vectors)
        annotations.append(ann)

    out = {key: coco[key] for key in coco if key != "annotations"}
    out["annotations"] = annotations
    out.update(arrays["meta"].get("extra_keys", {}))
    return out


def verify(annotated, coco, arrays):
    """Check that ``merge(coco, arrays)`` reproduces ``annotated`` for training.

    Does not build the merged dict (the train split is several GB). Returns
    a dict of findings; the first five must all be True:

    ``concepts_equal``           decoded pack == input concept vectors
    ``annotations_equal``        every input annotation, without its concept
                                 vector, equals the official one with that id,
                                 and both files have the same annotations
    ``image_fields_used_equal``  image id / file_name / width / height, in order
    ``categories_equal``
    ``extra_keys_stored``        non-standard top-level keys are in the pack
    ``metadata_diffs``           other differences (replaced by the official
                                 2017 values when merging)
    """
    anns = annotated["annotations"]
    dense = np.asarray([a["concept_vector"] for a in anns if "concept_vector" in a], dtype=np.uint8)
    findings = {"concepts_equal": bool(np.array_equal(unpack_concepts(arrays), dense))}
    del dense

    by_id = {a["id"]: a for a in coco["annotations"]}
    same = len(anns) == len(by_id)
    for a in anns:
        ref = by_id.get(a["id"])
        if ref is None or {k: v for k, v in a.items() if k != "concept_vector"} != ref:
            same = False
            break
    findings["annotations_equal"] = same

    a_img, b_img = annotated.get("images", []), coco.get("images", [])
    findings["image_fields_used_equal"] = (
        [tuple(i.get(k) for k in IMAGE_KEYS_USED) for i in a_img]
        == [tuple(i.get(k) for k in IMAGE_KEYS_USED) for i in b_img]
    )
    findings["categories_equal"] = annotated.get("categories") == coco.get("categories")
    findings["extra_keys_stored"] = arrays["meta"].get("extra_keys", {}) == {
        k: v for k, v in annotated.items() if k not in STANDARD_KEYS}

    diffs = {}
    for key in ("info", "licenses"):
        if annotated.get(key) != coco.get(key):
            diffs[key] = "differs"
    if findings["image_fields_used_equal"]:
        counts = {}
        for x, y in zip(a_img, b_img):
            for key in set(x) | set(y):
                if x.get(key) != y.get(key):
                    counts[key] = counts.get(key, 0) + 1
        diffs.update({f"images.{k}": f"{n} images" for k, n in counts.items()})
    order = [a["id"] for a in anns] != [a["id"] for a in coco["annotations"]]
    if order:
        diffs["annotation order"] = "differs (stored in the pack)"
    findings["metadata_diffs"] = diffs
    return findings
