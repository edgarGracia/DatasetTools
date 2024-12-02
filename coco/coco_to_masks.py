import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm
from PIL import Image
import argparse
from pycocotools.coco import COCO


def coco_to_mask(annots_path: Path, out_path: Path,
                 class_multiplier: int = 1, remap_class: bool = False,
                 remap_id: bool = False, add_id: bool = False,
                 i32: bool = False, binary: bool = False):

    dataset = COCO(annots_path)

    out_path.mkdir(exist_ok=True, parents=True)

    cats_ids = sorted(dataset.getCatIds())
    imgs_ids = sorted(dataset.getImgIds())

    if remap_class:
        remap_dict = {c: i+1 for i, c in enumerate(cats_ids)}

    for img_id in tqdm(imgs_ids):
        img_data = dataset.loadImgs([img_id])[0]
        img_h = img_data["height"]
        img_w = img_data["width"]
        img_name = img_data["file_name"]

        if i32:
            img = np.zeros((img_h, img_w), dtype=np.int32)
        else:
            img = np.zeros((img_h, img_w, 3), dtype="uint8")

        anns_ids = sorted(dataset.getAnnIds(imgIds=[img_id]))
        anns = dataset.loadAnns(anns_ids)
        for i, ann in enumerate(anns):
            id = ann["id"]
            cat = ann["category_id"]
            mask = dataset.annToMask(ann)
            if remap_class:
                cat = remap_dict[cat]
            if remap_id:
                id = i
            if binary:
                value = 255
            else:
                value = cat * class_multiplier
                value = value + id if add_id else value
            img[mask > 0] = value
        img = Image.fromarray(img)
        img.save(out_path.joinpath(img_name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Save mask images from a COCO dataset",
    )
    parser.add_argument(
        "annotations",
        type=Path,
        help="COCO annotations path"
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=None,
        help="Path to save the images"
    )
    parser.add_argument(
        "--class-multiplier",
        default=1,
        type=int,
        help="Class id multiplier"
    )
    parser.add_argument(
        "--remap-class",
        action="store_true",
        help="Remap class ids to incremental integers. e.g. 36,50 -> 0,1"
    )
    parser.add_argument(
        "--remap-id",
        action="store_true",
        help="Remap instance ids to incremental integers for each image"
    )
    parser.add_argument(
        "--add-id",
        action="store_true",
        help="Add the instance id to the class id"
    )
    parser.add_argument(
        "--i32",
        action="store_true",
        help="Save 2D int32 images"
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Save binary masks"
    )

    args = parser.parse_args()

    coco_to_mask(
        annots_path=args.annotations,
        out_path=args.output_path,
        class_multiplier=args.class_multiplier,
        remap_class=args.remap_class,
        remap_id=args.remap_id,
        add_id=args.add_id,
        i32=args.i32,
        binary=args.binary
    )
