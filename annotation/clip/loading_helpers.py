"""Descriptor loading / formatting helpers.

Adapted from ``loading_helpers.py`` and ``descriptor_strings.py`` of
Menon & Vondrick, "Visual Classification via Description from Large Language
Models" (ICLR 2023), https://github.com/sachit-menon/classify_by_description_release
(see NOTICE in this folder). Only the functions used by the COCO pipeline are kept.
"""

import json
import os
import random


def load_json(filename):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)


def stringtolist(description):
    """Parse an LLM answer into the list of its ``- item`` lines."""
    return [descriptor[2:] for descriptor in description.split('\n') if (descriptor != '') and (descriptor.startswith('- '))]


def wordify(string):
    return string.replace('_', ' ')


def make_descriptor_sentence(descriptor):
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"


def modify_descriptor(descriptor, apply_changes):
    if apply_changes:
        return make_descriptor_sentence(descriptor)
    return descriptor


def load_gpt_descriptions(hparams, classes_to_load=None):
    """Load ``{class: [descriptors]}`` and turn every descriptor into a CLIP prompt.

    With ``category_name_inclusion == 'prepend'`` (the COCO setting) a
    descriptor ``d`` of class ``k`` becomes
    ``"{before_text}{k}{between_text}{modify_descriptor(d)}{after_text}"``,
    e.g. ``"a photo of person, which has hair on head."``.

    Returns ``(gpt_descriptions, unmodify_dict)`` where ``unmodify_dict[k]``
    maps each prompt back to the raw descriptor.
    """
    gpt_descriptions_unordered = load_json(hparams['descriptor_fname'])
    unmodify_dict = {}

    if classes_to_load is not None:
        gpt_descriptions = {c: gpt_descriptions_unordered[c] for c in classes_to_load}
    else:
        gpt_descriptions = gpt_descriptions_unordered
    if hparams['category_name_inclusion'] is not None:
        if classes_to_load is not None:
            keys_to_remove = [k for k in gpt_descriptions.keys() if k not in classes_to_load]
            for k in keys_to_remove:
                print(f"Skipping descriptions for \"{k}\", not in classes to load")
                gpt_descriptions.pop(k)

        for i, (k, v) in enumerate(gpt_descriptions.items()):
            if len(v) == 0:
                v = ['']

            word_to_add = wordify(k)

            if (hparams['category_name_inclusion'] == 'append'):
                build_descriptor_string = lambda item: f"{modify_descriptor(item, hparams['apply_descriptor_modification'])}{hparams['between_text']}{word_to_add}"  # noqa: E731
            elif (hparams['category_name_inclusion'] == 'prepend'):
                build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{modify_descriptor(item, hparams['apply_descriptor_modification'])}{hparams['after_text']}"  # noqa: E731
            else:
                build_descriptor_string = lambda item: modify_descriptor(item, hparams['apply_descriptor_modification'])  # noqa: E731

            unmodify_dict[k] = {build_descriptor_string(item): item for item in v}
            gpt_descriptions[k] = [build_descriptor_string(item) for item in v]

            if i == 0:
                print(f"\nExample description for class {k}: \"{gpt_descriptions[k][0]}\"\n")
    return gpt_descriptions, unmodify_dict


def seed_everything(seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
