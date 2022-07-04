from typing import Union
from pathlib import Path
from typing import List
from tqdm import tqdm
import seaborn as sns
import argparse
import cv2
import matplotlib.pyplot as plt


CV2_FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_COLOR = (255,255,255)
FONT_BG_COLOR = (0,0,0)
FONT_SCALE = 1
FONT_THICK = 2

BB_THICK = 2
BB_COLOR = (200,255,0)
SNS_PALETTE = None


def read_annotation(path: Path, is_xyxy: bool = False,
    separator: str = " ") -> List[dict]:

    lines = path.read_text().strip().split("\n")
    
    bbs = []
    for line in lines:
        bb = line.split(separator)
        
        if len(bb) == 5:
            label = bb[0]
            score = None
            xmin, ymin, xmax, ymax = [float(i) for i in bb[1:]]
        elif len(bb) == 6:
            label = bb[0]
            score, xmin, ymin, xmax, ymax = [float(i) for i in bb[1:]]
        else:
            print(f"{path} bad annotation format!!")
            return []
        
        if not is_xyxy:
            cx, cy, w, h = xmin, ymin, xmax, ymax
            xmin = cx - w/2
            ymin = cy - h/2
            xmax = cx + w/2
            ymax = cy + h/2

        bbs.append({
            "label": label, "score": score, "xmin": xmin, "ymin": ymin,
            "xmax": xmax, "ymax": ymax
        })

    return bbs


def plot_bb(images_path: Path, annotations_path: Path,
    out_path: Union[Path, None] = None, min_score: Union[float, None] = None,
    is_xyxy: bool = False, is_absolute: bool = False, separator: str = " ",
    show: bool = True, show_score: bool = True, show_label: bool = True,
    color: Union[tuple, None] = None, label_mapping: Union[Path, None] = None,
    bb_thickness: int = 2, only_labels: Union[list, None] = None):
    
    # Parse the label mapping file
    if label_mapping is not None:
        label_mapping = label_mapping.read_text().strip().split("\n")

    # Create the output path
    if out_path is not None:
        assert images_path != out_path
        out_path.mkdir(exist_ok=True, parents=True)

    for img_path in tqdm(list(images_path.iterdir())):
        img = cv2.imread(str(img_path))
        
        # Parse annotation
        annot_path = annotations_path.joinpath(img_path.stem + ".txt")
        if not annot_path.exists():
            print(f"{annot_path} does not exist!!")
            continue
        
        bbs = read_annotation(annot_path, is_xyxy, separator)
        
        # Draw bounding box
        for i, bb in enumerate(bbs):
            
            # Filter score
            if min_score is not None and bb["score"] is not None:
                if bb["score"] < min_score:
                    continue
            
            # Filter label
            if only_labels is not None and bb["label"] not in only_labels:
                continue

            if not is_absolute:
                xmin = int(bb["xmin"] * img.shape[1])
                ymin = int(bb["ymin"] * img.shape[0])
                xmax = int(min(bb["xmax"] * img.shape[1], img.shape[1]-1))
                ymax = int(min(bb["ymax"] * img.shape[0], img.shape[0]-1))
            else:
                xmin = int(bb["xmin"])
                ymin = int(bb["ymin"])
                xmax = int(min(bb["xmax"], img.shape[1]-1))
                ymax = int(min(bb["ymax"], img.shape[0]-1))
            
            if color is None:
                try:
                    bb_color = sns.color_palette(
                        SNS_PALETTE,
                        int(bb["label"])+1,
                    )[int(bb["label"])]
                    bb_color = [int(255 * i) for i in bb_color]
                except Exception as e:
                    print(e)
                    bb_color = BB_COLOR
            else:
                bb_color = color
            bb_color = tuple(reversed(bb_color))

            # Bounding box
            img = cv2.rectangle(
                img,
                (xmin, ymin),
                (xmax, ymax),
                color=bb_color,
                thickness=bb_thickness
            )

            # Draw label
            label = ""
            if show_label:
                label = bb["label"]
                if label_mapping is not None:
                    try:
                        label = label_mapping[int(bb["label"])]
                    except Exception as e:
                        print(e)
            if show_score and bb['score'] is not None:
                if show_label:
                    label += ": "
                label += str(round(bb['score'], 3))
            
            if label:
                (text_w, text_h), _ = cv2.getTextSize(
                    bb["label"],
                    CV2_FONT,
                    FONT_SCALE,
                    FONT_THICK
                )

                # TODO: check limits
                img = cv2.rectangle(
                    img,
                    (xmin, ymin - text_h),
                    (xmin + text_w, ymin),
                    FONT_BG_COLOR,
                    -1
                )
                img = cv2.putText(
                    img,
                    label,
                    (xmin, ymin),
                    CV2_FONT,
                    FONT_SCALE,
                    FONT_COLOR,
                    FONT_THICK,
                    cv2.LINE_AA
                )

        if show:
            plt.clf()
            plt.imshow(img[:,:,::-1])
            plt.show()
        if out_path is not None:
            cv2.imwrite(str(out_path.joinpath(img_path.name)), img)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="""
    Visualize the bounding boxes from txt annotations.
    There must be an annotation file for each image with the same name and
    with the following format:
    
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
        "--output-path",
        type=Path,
        default=None,
        help="Path to save the images with the bounding boxes"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="The minimum score the show a bounding box"
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
        help=("""Coordinates are absolute. If not set coordinates must be
            relative to the image size""")
    )
    parser.add_argument(
        "--separator",
        default=" ",
        help="Data separator. Default to ' '"
    )
    parser.add_argument(
        "--color",
        default=None,
        help="""Set the same color for all the objects.
            RGB values separated by ',' e.g. '50,100,255'"""
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help=("Show the images on screen")
    )
    parser.add_argument(
        "--show-score",
        action="store_true",
        help=("Show the score value")
    )
    parser.add_argument(
        "--show-label",
        action="store_true",
        help=("Show the object label")
    )
    parser.add_argument(
        "--label-mapping",
        type=Path,
        help=("""File with a label name per line to translate labels from
            label-id to label-name""")
    )
    parser.add_argument(
        "--box-thickness",
        type=int,
        default=BB_THICK,
        help=("Bounding Box thickness. Default 2.")
    )
    parser.add_argument(
        "--only-labels",
        help=("Specify the labels to parse separated by ','")
    )
    
    args = parser.parse_args()

    color = [
        int(i) for i in args.color.split(",")
     ] if args.color is not None else None

    only_labels = [
        i for i in args.only_labels.split(",")
     ] if args.only_labels is not None else None

    plot_bb(
        images_path = args.images,
        annotations_path = args.annotations,
        out_path = args.output_path,
        min_score = args.min_score,
        is_xyxy = args.xyxy,
        is_absolute = args.is_absolute,
        separator = args.separator,
        show = args.show,
        show_score = args.show_score,
        show_label = args.show_label,
        color = color,
        label_mapping = args.label_mapping,
        bb_thickness =args.box_thickness,
        only_labels = only_labels
    )
