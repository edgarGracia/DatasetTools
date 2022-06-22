from pycocotools.coco import COCO
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2


def crop_coco(dataset_path: Path, images_path: Path, out_path: Path,
    separate_classes: bool = False, use_seg: bool = False):

    assert images_path != out_path
    out_path.mkdir(exist_ok=True, parents=True)

    dataset = COCO(dataset_path)
    images = dataset.imgs
    categories = dataset.cats
    
    for img_id, img_data in tqdm(images.items()):
        img_path = images_path.joinpath(img_data["file_name"])
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"Error reading {img_path}")
            continue
            
        annots = dataset.loadAnns(dataset.getAnnIds(imgIds=[img_id]))
        for i, annot in enumerate(annots):
            if use_seg:
                mask = dataset.annToMask(annot)

            xmin = max(0, int(annot["bbox"][0]))
            ymin = max(0, int(annot["bbox"][1]))
            xmax = min(int(annot["bbox"][2] + xmin), img.shape[1])
            ymax = min(int(annot["bbox"][3] + ymin), img.shape[0])

            crop = img.copy()[ymin:ymax, xmin:xmax, :]
            if use_seg:
                mask = mask[ymin:ymax, xmin:xmax]
                crop[mask == 0] = 0

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

    crop_coco(
        dataset_path=args.dataset,
        images_path=args.images,
        out_path=args.output_path,
        separate_classes=args.separate_classes,
        use_seg=args.use_seg
    )
