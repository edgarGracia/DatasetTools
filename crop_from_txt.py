from pathlib import Path
from typing import List
from tqdm import tqdm
import argparse
import cv2



def read_annotation(path: Path, is_xywh: bool = False,
    separator: str = " ",) -> List[dict]:

    with open(path, "r") as f:
        lines = [i.strip() for i in f.readlines()]
    
    bbs = []
    for line in lines:
        bb = line.split(separator)
        
        label = bb[0]
        xmin, ymin, xmax, ymax = [int(i) for i in bb[-4:]]
        
        if is_xywh:
            xmax += xmin
            ymax += ymin

        bbs.append({
            "class": label, "xmin": xmin, "ymin": ymin,
            "xmax": xmax, "ymax": ymax
        })

    return bbs


def crop_txt(images_path: Path, annotations_path: Path, out_path: Path,
    separate_classes: bool = False, recursive: bool = False,
    is_xywh: bool = False, separator: str = " "):
    
    assert images_path != out_path

    out_path.mkdir(exist_ok=True, parents=True)

    for img_path in tqdm(list(images_path.iterdir())):
        
        if img_path.is_dir():
            crop_txt(img_path, annotations_path.joinpath(img_path.name),
                out_path.joinpath(img_path.name), separate_classes, recursive,
                is_xywh, separator)
            continue

        annot_path = annotations_path.joinpath(img_path.stem + ".txt")
        
        try:
            bbs = read_annotation(annot_path, is_xywh, separator)
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
    Crop images with the bounding boxes on txt files.
    Annotations should have the following format:
    
    <label> [<score>] <xmin> <ymin> <xmax|width> <ymax|height>
    <label> [<score>] <xmin> <ymin> <xmax|width> <ymax|height>
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
        "--xywh",
        action="store_true",
        help=("Bounding box format is 'xmin ymin width height'\n" + 
        "Default is 'xmin ymin xmax ymax'")
    )
    parser.add_argument(
        "--separator",
        default=" ",
        help="Data seperator. Default to ' '"
    )
    args = parser.parse_args()

    crop_txt(args.images, args.annotations, args.output_path,
        args.separate_classes, args.recursive, is_xywh=args.xywh,
        separator=args.separator)
