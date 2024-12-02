from ast import parse
from typing import Union
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm
import seaborn as sns
import json
import argparse
import cv2
import matplotlib.pyplot as plt
import pycocotools.mask as Mask
from pycocotools.coco import COCO


DRAW_IMG = False

DRAW_SEG = True
SEG_ALPHA = 1.0

DRAW_BB = False
BB_COLOR = (255,200,255)
BB_THICKNESS = 3

DRAW_LABEL = False
LABEL_SCALE = 1
LABEL_BG_COLOR = (0,0,0)
LABEL_THICKNESS = 2
LABEL_BORDER = 2
CV2_FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_COLOR = (255,255,255)
FONT_BG_COLOR = (0,0,0)
LABEL_FG_COLOR = (255,255,255)

DRAW_POS = False
POSITION_SIZE = 3
POSITION_BORDER = 1
POSITION_FG_COLOR = (255,255,255)
POSITION_BG_COLOR = (0,0,0)



def _parse_coco(coco_path: Path) -> dict:
    return json.loads(coco_path.read_text())


def _draw_instances(img_path: Path(), instances: List[dict],
    out_path: Path = None, show: bool = False):
    
    img = cv2.imread(str(img_path))
    
    if not DRAW_IMG:
        img = np.zeros_like(img)

    for ins in instances:
        box = ins["bbox"]
        # xmin, ymin, xmax, ymax
        box = (
            int(box[0]),
            int(box[1]),
            int(box[2]) + int(box[0]),
            int(box[3]) + int(box[1]),
        )
        cx, cy = int((box[0] + box[2])/2), int((box[1] + box[3])/2)
        
        if isinstance(ins["segmentation"], dict):
            mask = Mask.decode(ins["segmentation"])
        else:
            rles = Mask.frPyObjects(
                ins["segmentation"],
                img.shape[0],
                img.shape[1]
            )
            rle = Mask.merge(rles)
            mask = Mask.decode(rle)

        category_id = ins["category_id"]

        # Draw segmentation
        if DRAW_SEG:
            color = sns.color_palette(None, category_id+1)[category_id]
            # color = sns.color_palette(None, ins["id"]+1)[ins["id"]]
            color_mask = np.full_like(img, [int(i*255) for i in color])
            img[mask!=0] = (
                img[mask!=0] * (1 - SEG_ALPHA)
            ) + (color_mask[mask!=0] * SEG_ALPHA)

            if DRAW_BB:
                img = cv2.rectangle(
                    img,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    color=BB_COLOR,
                    thickness=BB_THICKNESS
                )
            
            if DRAW_LABEL:
                text = f"{category_id}"
                txt_pos_x = cx
                txt_pos_y = cy
                img = cv2.putText(
                    img,
                    text,
                    (txt_pos_x, txt_pos_y),
                    CV2_FONT,
                    LABEL_SCALE,
                    LABEL_BG_COLOR,
                    LABEL_THICKNESS+LABEL_BORDER,
                    cv2.LINE_AA
                )
                img = cv2.putText(
                    img,
                    text,
                    (txt_pos_x, txt_pos_y),
                    CV2_FONT,
                    LABEL_SCALE,
                    LABEL_FG_COLOR,
                    LABEL_THICKNESS,
                    cv2.LINE_AA
                )
            
            if DRAW_POS:
                img = cv2.circle(
                    img,
                    (cx, cy),
                    POSITION_SIZE+POSITION_BORDER,
                    POSITION_BG_COLOR,
                    -1,
                    1
                )
                img = cv2.circle(
                    img,
                    (cx, cy),
                    POSITION_SIZE,
                    POSITION_FG_COLOR,
                    -1,
                    1
                )
    if show:
        plt.imshow(img[:,:,::-1])
        plt.show()
    if out_path is not None:
        cv2.imwrite(str(out_path.joinpath(img_path.name)), img)


def visualize_coco(images_path: Path, annots_path: Path,
    out_path: Path = None, show: bool = False):

    if out_path is not None:
        assert ((images_path != out_path or images_path.is_file()) and
            (images_path.parent != out_path or images_path.is_dir()))
        out_path.mkdir(exist_ok=True, parents=True)

    images_list = (
        [images_path] if images_path.is_file() else
        list(images_path.iterdir())
    )
    
    if annots_path.is_file():
        # Dataset json
        dataset = COCO(annots_path)
        images_id = dataset.getImgIds()
        images_names = [i.name for i in images_list]
        coco_images = [
            i for i in dataset.loadImgs(images_id)
            if i["file_name"] in images_names
        ]
        for coco_img in tqdm(coco_images):
            anns_ids = dataset.getAnnIds(imgIds=[coco_img["id"]])
            anns = dataset.loadAnns(anns_ids)
            _draw_instances(
                images_list[images_names.index(coco_img["file_name"])],
                anns,
                out_path,
                show
            )
    else:
        # Results json
        annots_dict = {
            i: annots_path.joinpath(i.stem+".json")
            for i in images_list
            if annots_path.joinpath(i.stem+".json").exists()
        }
        for image_path, annot_path in tqdm(annots_dict.items()):
            instances = _parse_coco(annot_path)
            _draw_instances(image_path, instances, out_path, show)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="""
    Visualize coco instances from a file or directory.
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
        "--output-path",
        type=Path,
        default=None,
        help="Path to save the images"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help=("Show the images")
    )
    
    args = parser.parse_args()

    visualize_coco(
        images_path = args.images,
        annots_path = args.annotations,
        out_path = args.output_path,
        show = args.show,
    )
