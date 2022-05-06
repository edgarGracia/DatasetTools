from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import cv2


def print_stats(stats):
    print(f"Total images: {stats['count_img']}")
    print(f"Mean (RGBA): {', '.join([str(i) for i in stats['mean']])}")
    print(f"STD  (RGBA): {', '.join([str(i) for i in stats['std']])}")
    print(f"Mean size (W x H): " +
        f"{', '.join([str(i.round(3)) for i in stats['img_sizes'].mean(0)])}")
    print(f"Max W: {stats['img_sizes'][:,1].max()}, " +
        f"Max H: {stats['img_sizes'][:,0].max()}")
    print(f"Min W: {stats['img_sizes'][:,1].min()}, " +
        f"Min H: {stats['img_sizes'][:,0].min()}")


def get_img_stats(img_root_path: Path, normalize: bool = False,
    recursive: bool = False) -> dict:
    
    count_img = 0
    count_pix = 0
    sum_ch = np.zeros((4))
    sum_std = np.zeros((4))
    img_sizes = []

    if recursive:
        img_paths = [i for i in img_root_path.rglob("*") if i.is_file()]
    else:
        img_paths = [i for i in img_root_path.iterdir() if i.is_file()]

    pbar = tqdm(total=len(img_paths))

    # Copute mean
    for img_path in img_paths[::-1]:
        # Read image
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Unable to read {img_path}")
            img_paths.remove(img_path)
            pbar.update(1)
            continue
        pbar.update(0.5)
        
        img = np.stack((img,)*4, axis=-1) if img.ndim == 2 else img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA) if img.shape[-1] == 3 else img
        count_img += 1
        count_pix += (img.shape[0] * img.shape[1])
        sum_ch += img.sum((0,1))
        img_sizes.append(np.array(img.shape[:2]))
    
    if count_img != 0:
        mean = sum_ch/count_pix
    
        # Compute std
        for img_path in img_paths:
            pbar.update(0.5)
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            img = np.stack((img,)*4, axis=-1) if img.ndim == 2 else img
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA) if img.shape[-1] == 3 else img
            sum_std += ((img - mean)**2).sum((0,1))
        pbar.close()
        std = (sum_std / count_pix)**0.5

        # BGRA to RGBA
        mean = mean[[2,1,0,3]]
        std = std[[2,1,0,3]]

        if normalize:
            mean = mean / 255
            std = std / 255

        mean = np.around(mean, 3)
        std = np.around(std, 3)
        img_sizes = np.array(img_sizes)
        
        return {
            "count_img": count_img,
            "mean": mean,
            "std": std,
            "img_sizes": img_sizes
        }

    return None  


def get_img_stats_one_pass(img_root_path: Path, normalize: bool = False,
    recursive: bool = False) -> dict:
    
    count_img = 0
    count_pix = 0
    old_mean = np.zeros((4))
    mean = np.zeros((4))
    old_std = np.zeros((4))
    std = np.zeros((4))
    img_sizes = []

    if recursive:
        img_paths = [i for i in img_root_path.rglob("*") if i.is_file()]
    else:
        img_paths = [i for i in img_root_path.iterdir() if i.is_file()]

    for img_path in tqdm(img_paths):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Unable to read {img_path}")
            continue
        
        img = np.stack((img,)*4, axis=-1) if img.ndim == 2 else img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA) if img.shape[-1] == 3 else img
        
        count_img += 1
        img_sizes.append(np.array(img.shape[:2]))

        # TODO: vectorize
        for i, p in enumerate(img.reshape(img.shape[0] * img.shape[1], 4)):
            count_pix += 1

            if count_pix == 1:
                old_mean = mean = p
                old_std = 0

            mean = old_mean + (p - old_mean) / count_pix
            std = old_std + (p - old_mean) * (p - mean)
            
            old_mean = mean
            old_std = std
               
    if count_pix != 0:
        std = np.sqrt(std / (count_pix - 1))

        # BGRA to RGBA
        mean = mean[[2,1,0,3]]
        std = std[[2,1,0,3]]

        if normalize:
            mean = mean / 255
            std = std / 255

        mean = np.around(mean, 3)
        std = np.around(std, 3)
        img_sizes = np.array(img_sizes)
        
        return {
            "count_img": count_img,
            "mean": mean,
            "std": std,
            "img_sizes": img_sizes
        }

    return None  



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="""Calculate the mean, std
    and size of all the images on a directory""")
    
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the images folder"
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Traverse all the folder on the input path"
    )
    parser.add_argument(
        "-n",
        "--normalize",
        action="store_true",
        help="Normalize values"
    )
    
    args = parser.parse_args()
    
    stats = get_img_stats_one_pass(args.input_path, args.normalize, args.recursive)
    if stats is not None:
        print_stats(stats)

    stats = get_img_stats(args.input_path, args.normalize, args.recursive)
    if stats is not None:
        print_stats(stats)
