"""Ask an OpenAI chat model for 10 visual descriptors per COCO class.

Writes ``{class_name: [descriptor, ...]}`` (saved after every batch of
classes). The released file is ``concepts/per_class_descriptors_10.json``;
``concepts/per_class_descriptors_top5.json`` keeps 5 descriptors per class
(selected by hand; for 40 classes at least one descriptor was edited or merged).

The API key is read from the ``OPENAI_API_KEY`` environment variable only.
Prompt format adapted from Menon & Vondrick (ICLR 2023), see NOTICE.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loading_helpers import stringtolist  # noqa: E402
from common.coco_classes import COCO_CLASSES  # noqa: E402


def generate_prompt(category_name: str):
    return f"""Q: What are the 10 most useful visual characteristics for distinguishing a lemur in a photo?
    A: These are the 10 most useful visual features to tell there is a lemur in a photo:
    - four-limbed primate
    - black, grey, white, brown, or red-brown
    - wet and hairless nose with curved nostrils
    - Long, bushy tail
    - large eyes
    - slender limbs
    - furry bodies
    - clawed hands and feet
    - arboreal posture
    - pointed snout

    Q: What are the 10 most useful visual characteristics for distinguishing a television in a photo?
    A: These are the 10 most useful visual features to tell there is a television in a photo:
    - electronic device
    - thin bezel/frame surrounding the display
    - glossy or reflective glass surface
    - black, grey or dark-colored
    - a large, rectangular screen
    - a stand or mount to support the screen
    - visible support feet
    - one or more speakers
    - a power cord
    - input ports for connecting to other devices

    Q: What are the 10 most useful visual characteristics for distinguishing a {category_name} in a photo?
    A: These are the 10 most useful visual features to tell there is a {category_name} in a photo:
    -
    """


def extract_text_from_response(resp):
    """Return the text of a Responses API result (``output_text`` or walking ``output``)."""
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    texts = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if isinstance(c, str):
                texts.append(c)
            elif isinstance(c, dict):
                t = c.get("text") or c.get("content") or ""
                if t:
                    texts.append(t)
    return "\n".join(texts).strip()


def obtain_descriptors_and_save(filename, class_list, batch_size=20, model="gpt-3.5-turbo", retries=3):
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set the OPENAI_API_KEY environment variable.")
    client = OpenAI(api_key=api_key)

    if not filename.endswith('.json'):
        filename += '.json'
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    prompts = [generate_prompt(category.replace('_', ' ')) for category in class_list]
    descriptors = {}
    for start in range(0, len(prompts), batch_size):
        for prompt, cat in zip(prompts[start:start + batch_size], class_list[start:start + batch_size]):
            text = ""
            for attempt in range(retries):
                try:
                    resp = client.responses.create(model=model, input=prompt, temperature=0.0, max_output_tokens=400)
                    text = getattr(resp, "output_text", None) or extract_text_from_response(resp)
                    if text and text.strip():
                        break
                except Exception as e:
                    print(f"Error for {cat} (attempt {attempt + 1}): {e}")
            if not text:
                print(f"WARNING: empty answer for class {cat}")
                descriptors[cat] = []
                continue
            descriptors[cat] = stringtolist(text)

        # Save after every batch so partial results survive failures.
        with open(filename, 'w', encoding='utf-8') as fp:
            json.dump(descriptors, fp, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", required=True, help="Output JSON, e.g. descriptors_10.json.")
    parser.add_argument("--classes", nargs="+", default=None, help="Class names (default: the 80 COCO classes).")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model (default: gpt-3.5-turbo).")
    parser.add_argument("--batch_size", type=int, default=20, help="Classes between intermediate saves.")
    args = parser.parse_args()
    obtain_descriptors_and_save(args.output, args.classes or COCO_CLASSES, args.batch_size, args.model)


if __name__ == "__main__":
    main()
