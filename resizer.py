from pathlib import Path
from tqdm import tqdm
import argparse
import math
import cv2

MIN_LEADING_ZEROS = 8



def resize_images(path_in: Path, path_out: Path, factor: float = None,
    width: int = None, height: int = None, keep_ar: bool = False, 
    suffix: str = None, rename: bool = False) -> list:
    
    assert path_in != path_out

    # Create output path
    path_out.mkdir(exist_ok=True, parents=True)

    # Get images paths
    if path_in.is_file():
        images_paths = [path_in]
    else:
        images_paths = list(path_in.iterdir())
    
    if rename:
        counter = 0
        num_imgs = len([i for i in images_paths if i.is_file()])
        leading_zeros = (MIN_LEADING_ZEROS if num_imgs < 10**MIN_LEADING_ZEROS
            else math.ceil(math.log10(num_imgs)))

    dirs_list = []
    for img_path in tqdm(images_paths):
        # Read image
        if img_path.is_dir():
            dirs_list.append(img_path)
            continue
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                print(f"Error reading {img_path}")
                continue
        
        # Calculate size
        dsize = None
        if factor is None:
            if keep_ar:
                if width is not None:
                    if height is not None:
                        dsize = (width, height)    # w!=None & h!=None
                    else:                          # w!=None & h==None
                        factor = width / img.shape[1]
                elif height is not None:           # w==None & h!=None
                    factor = height / img.shape[0]
            else:
                w = width if width is not None else img.shape[1]
                h = height if height is not None else img.shape[0]
                dsize = (w, h)

        # Resize
        if dsize is not None or factor is not None:
            img = cv2.resize(img, dsize, fx=factor, fy=factor)
        
        # Set name
        if rename:
            name = str(counter).zfill(leading_zeros)
            suffix = suffix if suffix is not None else img_path.suffix
            name += suffix
            counter += 1
        else:
            name = img_path.name if suffix is None else img_path.stem + suffix
        
        # Save
        cv2.imwrite(str(path_out.joinpath(name)), img)
    return dirs_list


def run_recursive(path_in: Path, path_out: Path, factor: float = None,
    width: int = None, height: int = None, keep_ar: bool = False,
    suffix: str = None, rename: bool = False):

    path_out.mkdir(exist_ok=True, parents=True)
    dirs = [path_in]
    while dirs:
        dir = dirs.pop()
        out = Path(str(dir).replace(str(path_in), str(path_out)))
        dirs += resize_images(dir, out, factor,
            width, height, keep_ar, suffix, rename)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="""Resize and/or change the
    format of all the images on a folder.""")
    
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
        "-s",
        "--suffix",
        type=str,
        required=False,
        help="Optional image extension e.g. .jpg"
    )
    parser.add_argument(
        "-W",
        "--width",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-f",
        "--factor",
        type=float,
        required=False,
        help="Resize factor e.g. 0.5"
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Traverse all the folder on the input-path"
    )
    parser.add_argument(
        "-k",
        "--keep-aspect-ratio",
        dest="keep_ar",
        action="store_true",
        help="Mantain the aspect ratio of the images"
    )
    parser.add_argument(
        "--rename",
        action="store_true",
        help="Rename the images with a counter" 
    )
    parser.add_argument(
        "--zeros",
        type=int,
        default=MIN_LEADING_ZEROS,
        help=f"Minimum number of leading zeros when --rename is set. Default {MIN_LEADING_ZEROS}"
    )
    
    args = parser.parse_args()
    
    MIN_LEADING_ZEROS = args.zeros

    if args.recursive:
        run_recursive(args.input_path, args.output_path, args.factor,
            args.width, args.height, args.keep_ar, args.suffix, args.rename)
    else:
        resize_images(args.input_path, args.output_path, args.factor,
            args.width, args.height, args.keep_ar, args.suffix, args.rename)