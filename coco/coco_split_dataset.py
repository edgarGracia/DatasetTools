from pathlib import Path
import numpy as np
import argparse
import random
import math
import json



def save_coco(path: Path, images: list, categories: list, annotations: list):
    with open(path, 'wt', encoding='UTF-8') as f:
        json.dump({
            'images': images,
            'categories': categories,
            'annotations': annotations
        }, f)


def split_coco(dataset_path: Path, train_split: float, val_split: float,
    test_split: float, out_train: Path, out_val: Path, out_test: Path,
    shuffle: bool = False):

    # Read coco json
    with open(dataset_path) as f:
        data = json.load(f)

    data_images = data["images"]
    data_annots = data["annotations"]
    data_cats = data["categories"]

    # Split images
    indexes = list(range(len(data_images)))
    if shuffle:
        random.shuffle(indexes)
    
    train_end = int(len(indexes) * train_split)
    val_end = train_end + math.ceil(len(indexes) * val_split)
    train_indexes = indexes[:train_end]
    val_indexes = indexes[train_end:val_end]
    test_indexes = indexes[val_end:]

    images_train = [data_images[i] for i in train_indexes]
    images_val = [data_images[i] for i in val_indexes]
    images_test = [data_images[i] for i in test_indexes]
    
    # Split annotations
    train_img_ids = [i["id"] for i in images_train]
    val_img_ids = [i["id"] for i in images_val]
    test_img_ids = [i["id"] for i in images_test]

    train_annots = [d for d in data_annots if d["image_id"] in train_img_ids]
    val_annots = [d for d in data_annots if d["image_id"] in val_img_ids]
    test_annots = [d for d in data_annots if d["image_id"] in test_img_ids]
    
    # Save the new json datasets
    save_coco(out_train, images_train, data_cats, train_annots)
    if out_val is not None:
        save_coco(out_val, images_val, data_cats, val_annots)
    if out_test is not None:
        save_coco(out_test, images_test, data_cats, test_annots)
    
    print(f"Train: {len(images_train)}, Val: {len(images_val)}, Test: {len(images_test)}")



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Split a coco dataset")
    
    argparser.add_argument(
        'dataset',
        help="Path to the json dataset",
        type=Path
    )
    argparser.add_argument(
        '--train',
        help="Train split [0, 1.]",
        required=True,
        type=float
    )
    argparser.add_argument(
        '--test',
        help="Optional test split [0, 1.]",
        default=0.,
        type=float
    )
    argparser.add_argument(
        '--val',
        help="Optional val split [0, 1.]",
        default=0.,
        type=float
    )
    argparser.add_argument(
        '--out-train',
        help="Output train json",
        required=True,
        type=Path
    )
    argparser.add_argument(
        '--out-test',
        help="Optional test output json",
        required=False,
        type=Path
    )
    argparser.add_argument(
        '--out-val',
        help="Optional validation output json",
        required=False,
        type=Path
    )
    argparser.add_argument(
        '--shuffle',
        help="Shuffle the images",
        action="store_true"
    )
    argparser.add_argument(
        '--seed',
        help="Random seed",
        required=False,
        type=float
    )

    args = argparser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.val == 0. and args.test == 0.:
        args.test = 1 - args.train
    np.testing.assert_almost_equal(args.train + args.val + args.test, 1.0)

    split_coco(args.dataset, args.train, args.val, args.test,
        args.out_train, args.out_val, args.out_test, args.shuffle)
