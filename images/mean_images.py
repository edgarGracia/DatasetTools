import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def mean_image(images_path: Path, output_path: Path):

    dtype = "float64"
    out_img = None
    n = 0

    try:
        for img_path in tqdm(list(images_path.iterdir())):
            try:
                img = cv2.imread(str(img_path))
                assert img is not None
            except KeyboardInterrupt:
                break
            except:
                continue

            n += 1
            if out_img is None:
                out_img = img.astype(dtype)
            else:
                out_img += img
    except KeyboardInterrupt:
        pass

    out_img /= n
    out_img = np.clip(out_img, 0, 255)
    out_img = out_img.astype("uint8")

    cv2.imwrite(
        str(output_path),
        out_img
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create a mean image from the "
                                 "images on directory")
    ap.add_argument(
        "images",
        help="Path to the images folder",
        type=Path
    )
    ap.add_argument(
        "output",
        help="Output image path",
        type=Path
    )

    args = ap.parse_args()
    mean_image(args.images, args.output)
