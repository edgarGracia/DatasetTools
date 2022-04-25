import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List
from tqdm import tqdm
import argparse
import cv2



def read_annotation(path: Path) -> List[dict]:
    tree = ET.parse(path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    bbs = []
    for obj in root.iter("object"):
        bb = {"class": obj.find("name").text}

        bndbox = obj.find("bndbox")
        bb["xmin"] = max(int(bndbox.find("xmin").text), 0)
        bb["xmax"] = min(int(bndbox.find("xmax").text), width)
        bb["ymin"] = max(int(bndbox.find("ymin").text), 0)
        bb["ymax"] = min(int(bndbox.find("ymax").text), height)

        bbs.append(bb)

    return bbs


def crop_voc(images_path: Path, annotations_path: Path, out_path: Path,
    separate_classes: bool = False, recursive: bool = False):
    
    assert images_path != out_path

    out_path.mkdir(exist_ok=True, parents=True)

    for img_path in tqdm(list(images_path.iterdir())):
        
        if img_path.is_dir():
            crop_voc(img_path, annotations_path.joinpath(img_path.name),
                out_path.joinpath(img_path.name), separate_classes, recursive)
            continue

        annot_path = annotations_path.joinpath(img_path.stem + ".xml")
        
        try:
            bbs = read_annotation(annot_path)
        except Exception as e:
            print(e)
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error reading {img_path}")
            continue
            
        for i, bb in enumerate(bbs):
            crop = img[bb["ymin"]:bb["ymax"], bb["xmin"]:bb["xmax"], :]

            out_name = f"{img_path.stem}_{i}{img_path.suffix}"
            if separate_classes:
                out_img_path = out_path.joinpath(bb["class"])
                out_img_path.mkdir(exist_ok=True)
                out_img_path = out_img_path.joinpath(out_name)
            else:
                out_img_path = out_path.joinpath(out_name)
            
            try:
                cv2.imwrite(str(out_img_path), crop)
            except Exception as e:
                print(out_img_path, e)
                continue



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="""
    Crop images with the bounding boxes of VOC annotations""")

    parser.add_argument(
        "images",
        type=Path,
        help="Images path"
    )
    parser.add_argument(
        "annotations",
        type=Path,
        help="Annotations path"
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
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Run recursively"
    )
    args = parser.parse_args()

    crop_voc(args.images, args.annotations, args.output_path,
        args.separate_classes, args.recursive)
