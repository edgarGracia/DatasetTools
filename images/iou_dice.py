from importlib.resources import path
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import cv2



def get_dice(img_a: np.ndarray, img_b: np.ndarray) -> float:
    intersection = np.logical_and(img_a, img_b)
    return (2*np.sum(intersection)) / ( np.sum(img_a) + np.sum(img_b) )


def get_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b)
    union = np.logical_or(mask_a, mask_b)
    return np.sum(intersection)/np.sum(union)


def get_mask(img_path: Path) -> np.ndarray:
    mask = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2GRAY) > 0
    return mask


def iou_and_dice_from_path(images_a: Path, images_b: Path) -> None:
    if images_a.is_file() and images_b.is_file():
        mask_a = get_mask(images_a)
        mask_b = get_mask(images_b)
        iou = get_iou(mask_a, mask_b)
        dice = get_dice(mask_a, mask_b)
        print(f"IOU: {iou}  |  DICE: {dice}")
    else:
        ious, dices = [], []
        for path_a in images_a.iterdir():
            path_b = images_b.joinpath(path_a.name)
            if  path_b.exists():
                mask_a = get_mask(path_a)
                mask_b = get_mask(path_b)
                iou = get_iou(mask_a, mask_b)
                dice = get_dice(mask_a, mask_b)
                ious.append(iou)
                dices.append(dice)
                print(f"{path_a.name}:  IOU: {iou}  |  DICE: {dice}")
        print(f"Mean IOU: {np.array(ious).mean()}  |  Mean DICE: {np.array(dices).mean()}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="""Calculate the intersection over union (IOU) and DICE 
        coefficient between two black and white images""")
    
    parser.add_argument(
        "image_a",
        type=Path,
        help="Path to an image or folder"
    )
    parser.add_argument(
        "image_b",
        type=Path,
        help="Path to an image or folder"
    )
    
    args = parser.parse_args()
    
    
    iou_and_dice_from_path(args.image_a, args.image_b)
