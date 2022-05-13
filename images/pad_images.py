from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import cv2


def pad_images(input_path: Path, output_path: Path, width: int, height: int,
    ignore_bigger: bool) -> list:
    
    assert input_path != output_path

    # Create output path
    output_path.mkdir(exist_ok=True, parents=True)

    # Get images paths
    if input_path.is_file():
        images_paths = [input_path]
    else:
        images_paths = list(input_path.iterdir())
    
    dirs_list = []
    for img_path in tqdm(images_paths):
        # Read image
        if img_path.is_dir():
            dirs_list.append(img_path)
            continue
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"\nError reading {img_path}")
                continue

        if len(img.shape) > 2:
            img_h, img_w, img_c = img.shape
            new_img = np.zeros((height, width, img_c), dtype=img.dtype)
        else:
            img_h, img_w = img.shape
            new_img = np.zeros((height, width), dtype=img.dtype)
        
        if ignore_bigger and (img_h > height or img_w > width):
            continue

        # Calculate indexes to center the image
        ny1 = max((height - img_h) // 2, 0)
        ny2 = min(ny1 + img_h, height)
        nx1 = max((width - img_w) // 2, 0)
        nx2 = min(nx1 + img_w, width)

        iy1 = max((img_h - height) // 2, 0)
        iy2 = min(iy1 + height, img_h)
        ix1 = max((img_w - width) // 2, 0)
        ix2 = min(ix1 + width, img_w)

        new_img[ny1:ny2, nx1:nx2] = img[iy1:iy2, ix1:ix2]

        cv2.imwrite(str(output_path / img_path.name), new_img)
    return dirs_list


def run_recursive(input_path: Path, output_path: Path, width: int, height: int,
    ignore_bigger: bool):

    output_path.mkdir(exist_ok=True, parents=True)
    dirs = [input_path]
    while dirs:
        dir = dirs.pop()
        out = Path(str(dir).replace(str(input_path), str(output_path)))
        dirs += pad_images(dir, out, width, height,
            ignore_bigger)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Adds padding to all the images on a folder.")
    
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to an image or folder"
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Output path where save the images"
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        help="Image output height",
        required=True,
    )
    parser.add_argument(
        "-W",
        "--width",
        type=int,
        help="Image output width",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Traverse all the folder on the input path"
    )
    parser.add_argument(
        "--ignore-bigger",
        action="store_true",
        help="Ignore images bigger than the target size. If not set images will be trimmed"
    )
    
    args = parser.parse_args()
    
    if args.recursive:
        run_recursive(args.input_path, args.output_path, args.width,
            args.height, args.ignore_bigger)
    else:
        pad_images(args.input_path, args.output_path, args.width, args.height,
            args.ignore_bigger)