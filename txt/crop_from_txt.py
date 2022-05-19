from pathlib import Path
from typing import List
from tqdm import tqdm
import argparse
import cv2



def read_annotation(path: Path, is_xyxy: bool = False,
    separator: str = " ",) -> List[dict]:

    lines = path.read_text().strip().split("\n")
    
    bbs = []
    for line in lines:
        bb = line.split(separator)
        
        label = bb[0]
        xmin, ymin, xmax, ymax = [float(i) for i in bb[-4:]]
        
        if not is_xyxy:
            cx, cy, w, h = xmin, ymin, xmax, ymax
            xmin = cx - w/2
            ymin = cy - h/2
            xmax = cx + w/2
            ymax = cy + h/2

        bbs.append({
            "class": label, "xmin": xmin, "ymin": ymin,
            "xmax": xmax, "ymax": ymax
        })

    return bbs


def crop_txt(images_path: Path, annotations_path: Path, out_path: Path,
    separate_classes: bool = False, is_xyxy: bool = False,
    is_absolute: bool = False, separator: str = " ", recursive: bool = False):
    
    assert images_path != out_path

    out_path.mkdir(exist_ok=True, parents=True)

    for img_path in tqdm(list(images_path.iterdir())):
        
        if img_path.is_dir():
            if recursive:
                crop_txt(img_path, annotations_path.joinpath(img_path.name),
                    out_path.joinpath(img_path.name), separate_classes,
                    is_xyxy, is_absolute, separator, recursive)
            continue

        annot_path = annotations_path.joinpath(img_path.stem + ".txt")
        
        try:
            bbs = read_annotation(annot_path, is_xyxy, separator)
        except Exception as e:
            print(e)
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error reading {img_path}")
            continue
            
        for i, bb in enumerate(bbs):
            if not is_absolute:
                bb["xmin"] = bb["xmin"] * img.shape[1]
                bb["ymin"] = bb["ymin"] * img.shape[0]
                bb["xmax"] = min(bb["xmax"] * img.shape[1], img.shape[1])
                bb["ymax"] = min(bb["ymax"] * img.shape[0], img.shape[0])
                
            crop = img[
                int(bb["ymin"]):int(bb["ymax"]),
                int(bb["xmin"]):int(bb["xmax"]),
                :
            ]

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
    Crop images with the bounding boxes on txt files.
    Annotations should have the following format:
    
    <label> [<score>] <x-cent|xmin> <y-cent|ymin> <width|xmax> <height|ymax>
    <label> [<score>] <x-cent|xmin> <y-cent|ymin> <width|xmax> <height|ymax>
    ...

    """, formatter_class=argparse.RawTextHelpFormatter)

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
    parser.add_argument(
        "--xyxy",
        action="store_true",
        help=("Bounding box format is '<xmin> <ymin> <xmax> <ymax>\n" +
            "Default is '<x-cent> <y-cent> <width> <height>'")
    )
    parser.add_argument(
        "--is-absolute",
        action="store_true",
        help=("Coordinates are absolute")
    )
    parser.add_argument(
        "--separator",
        default=" ",
        help="Data seperator. Default to ' '"
    )
    args = parser.parse_args()

    crop_txt(
        images_path = args.images,
        annotations_path = args.annotations,
        out_path = args.output_path,
        separate_classes = args.separate_classes,
        is_xyxy = args.xyxy,
        is_absolute = args.is_absolute,
        separator = args.separator,
        recursive = args.recursive
    )
