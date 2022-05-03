from pycocotools import mask
from typing import Tuple
from pathlib import Path
from typing import List
from tqdm import tqdm
import argparse
import json
import cv2



def read_coco(path: Path, class_names: list = None) -> Tuple[dict, dict, dict]:
    with open(path) as f:
        data = json.load(f)
    
    images = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat for cat in data["categories"]}
    annotations = {annot["id"]: annot for annot in data["annotations"]}

    # Filter classes
    if class_names is not None:
        cats_ids = [k for k,v in categories.items() if v["name"] in class_names]
        annotations = {annot["id"]: annot for annot in annotations.values()
            if annot["category_id"] in cats_ids}

    return images, categories, annotations


def crop_coco(images_path: Path, images_data: dict, categories: dict,
    annotations: dict, out_path: Path, separate_classes: bool = False,
    use_seg: bool = False):

    assert images_path != out_path

    out_path.mkdir(exist_ok=True, parents=True)

    annots_per_image = {img["file_name"]:
        [annot for annot in annotations.values() if annot["image_id"] == k]
        for k, img in images_data.items()
    }
    
    for img_name, annots in tqdm(annots_per_image.items()):
        if not annots:
            continue

        img_path = images_path / img_name
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"Error reading {img_path}")
            continue
            
        for i, annot in enumerate(annots):
            if use_seg:
                pass
            else:
                xmin = max(0, int(annot["bbox"][0]))
                ymin = max(0, int(annot["bbox"][1]))
                xmax = min(int(annot["bbox"][2] + xmin), img.shape[1])
                ymax = min(int(annot["bbox"][3] + ymin), img.shape[0])

                crop = img[ymin:ymax, xmin:xmax, :]

            out_name = f"{img_path.stem}_{i}{img_path.suffix}"
            if separate_classes:
                out_img_path = out_path.joinpath(
                    categories[annot["category_id"]]["name"])
                out_img_path.mkdir(exist_ok=True)
                out_img_path = out_img_path / out_name
            else:
                out_img_path = out_path / out_name
            
            try:
                cv2.imwrite(str(out_img_path), crop)
            except Exception as e:
                print(out_img_path, e)
                continue



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="""
    Crop images with the annotations of a COCO dataset""")

    parser.add_argument(
        "images",
        type=Path,
        help="Images path"
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Dataset path"
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to save the cropped images"
    )
    parser.add_argument(
        "--separate-classes",
        action="store_true",
        help="Separate images by its class on different folders"
    )
    # TODO: --use-seg
    parser.add_argument(
        "--use-seg",
        action="store_true",
        help="Crop the images with the segmentation data"
    )
    parser.add_argument(
        "--class-names",
        help="Optional list of class names to process. e.g. 'cat,dog'",
        required=False
    )
    args = parser.parse_args()

    class_names = (args.class_names.split(",") if args.class_names is not None
        else None)

    images_data, categories, annotations = read_coco(args.dataset, class_names)
    
    crop_coco(
        images_path=args.images,
        images_data=images_data,
        categories=categories,
        annotations=annotations,
        out_path=args.output_path,
        separate_classes=args.separate_classes,
        use_seg=args.use_seg
    )
