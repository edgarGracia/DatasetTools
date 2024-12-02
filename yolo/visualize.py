import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

COLOR = (255, 255, 255)
THICKNESS = 2


def draw_boxes(
    image_path: Path,
    boxes: list,
    out_path: Path | None,
    show: bool
):
    img = cv2.imread(str(image_path))
    h, w, _ = img.shape

    for c, xcent, ycent, width, height in boxes:
        xmin = int((xcent - width / 2) * w)
        xmax = int((xcent + width / 2) * w)
        ymin = int((ycent - height / 2) * h)
        ymax = int((ycent + height / 2) * h)
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), COLOR, THICKNESS)

    if show:
        plt.imshow(img[:, :, ::-1])
        plt.show()

    if out_path is not None:
        cv2.imwrite(str(out_path.joinpath(image_path.name)), img)


def read_annot(annots_path: Path, images_path: Path) -> list | None:
    annot_path = annots_path / f"{images_path.stem}.txt"
    if not annot_path.exists():
        print(f"no file {annot_path}")
        return None
    annots = annot_path.read_text().splitlines()
    boxes = []
    for a in annots:
        box = [float(i) for i in a.split(" ")]
        boxes.append(box)
    return boxes


def visualize_yolo(
    images_path: Path,
    annots_path: Path,
    out_path: Path = None,
    show: bool = False
):
    if out_path is not None:
        out_path.mkdir(exist_ok=True, parents=True)

    if images_path.is_file():
        boxes = read_annot(annots_path, images_path)
        draw_boxes(images_path, boxes, out_path, show)
    else:
        for img_path in tqdm(list(images_path.iterdir())):
            if img_path.suffix not in [".png", ".jpg"]:
                continue
            boxes = read_annot(annots_path, img_path)
            if boxes is None:
                continue
            draw_boxes(img_path, boxes, out_path, show)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "images",
        type=Path,
        help="Images path"
    )
    ap.add_argument(
        "annotations",
        type=Path,
        help="Annotations path"
    )
    ap.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=None,
        help="Path to save the images"
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Show the images"
    )
    args = ap.parse_args()

    visualize_yolo(
        images_path=args.images,
        annots_path=args.annotations,
        out_path=args.output_path,
        show=args.show,
    )
