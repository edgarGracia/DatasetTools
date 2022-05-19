import xml.etree.ElementTree as ET
from typing import Tuple, List
from pathlib import Path
from tqdm import tqdm
import argparse


def read_annotation(path: Path) -> Tuple[List[dict], int, int]:
    tree = ET.parse(path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    bbs = []
    for obj in root.iter("object"):
        bb = {"label": obj.find("name").text}

        bndbox = obj.find("bndbox")
        bb["xmin"] = max(int(bndbox.find("xmin").text), 0)
        bb["xmax"] = min(int(bndbox.find("xmax").text), width)
        bb["ymin"] = max(int(bndbox.find("ymin").text), 0)
        bb["ymax"] = min(int(bndbox.find("ymax").text), height)

        bbs.append(bb)

    return bbs, width, height


def voc_to_yolo(annotations_path: Path, output_path: Path,
    convert_labels: bool = False, labels_mapping: Path = None,
    xyxy: bool = False, absolute: bool = False):
    
    assert annotations_path != output_path
    output_path.mkdir(exist_ok=True, parents=True)
    
    if convert_labels:
        labels_id = {}
        if labels_mapping is not None:
            lines = labels_mapping.read_text().strip().split("\n")
            labels_id = {l: i for i, l in enumerate(lines)}

    for annot_file in tqdm(list(annotations_path.iterdir())):
        out_file = output_path.joinpath(annot_file.stem + ".txt")
        out_str = ""
        
        try:
            bbs, img_w, img_h = read_annotation(annot_file)
        except Exception as e:
            print(e)
            continue
        
        for bb in bbs:
            label = bb["label"]
            if convert_labels:
                if label not in labels_id:
                    labels_id[label] = len(labels_id)
                label = labels_id[label]
            if not xyxy:
                w = bb["xmax"] - bb["xmin"]
                h = bb["ymax"] - bb["ymin"]
                xc = bb["xmin"] + w/2
                yc = bb["ymin"] + h/2
            else:
                xc, yc, w, h = bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]
            
            if absolute:
                xc, yc = int(xc), int(yc)
            else:
                xc, w = xc/img_w, w/img_w
                yc, h = yc/img_h, h/img_h
            
            out_str += f"{label} {xc} {yc} {w} {h}\n"

        out_file.write_text(out_str)
    
    if convert_labels:
        id_labels = {v:k for k,v in labels_id.items()}
        for i in sorted(list(id_labels.keys())):
            print(id_labels[i])



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="""
    Convert VOC annotations to YOLO format.
    The output is a single .txt file for each image with one line
    per each object, with the following format:
    <object-class> <center-x> <center-y> <width> <height>.
    Coordinates are relative to the image size""")

    parser.add_argument(
        "annotations_path",
        type=Path,
        help="Path to the folder with the VOC annotations"
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Output YOLO annotations path"
    )
    parser.add_argument(
        "--xyxy",
        action="store_true",
        help="Change bounding box format to <xmin> <ymin> <xmax> <ymax>"
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Use absolute coordinates"
    )
    parser.add_argument(
        "--convert-labels",
        action="store_true",
        help="Automatically convert class names to integer numbers by its appearance order"
    )
    parser.add_argument(
        "--labels-mapping",
        type=Path,
        default=None,
        help="Path to the class names mapping file to use with --convert-labels. This should contain one class name per line."
    )
    args = parser.parse_args()

    voc_to_yolo(args.annotations_path, args.output_path, args.convert_labels,
        args.labels_mapping, args.xyxy, args.absolute)
