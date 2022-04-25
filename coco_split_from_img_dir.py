from pathlib import Path
import argparse
import json



def save_coco(path: Path, images: list, categories: list, annotations: list):
    with open(path, 'wt', encoding='UTF-8') as f:
        json.dump({
            'images': images,
            'categories': categories,
            'annotations': annotations
        }, f)


def _filter(data: list, ids: list, key: str):
    return [d for d in data if d[key] in ids]


def _get_ids(images_path: Path, images_ids: dict):
    ids = []
    for img_path in images_path.iterdir():
        try:
            ids.append(images_ids[img_path.name])
        except KeyError:
            print(f"image {img_path.name} not found")
            continue
    return ids


def _check_intersection(path_1: Path, ids_1: list, path_2: Path, ids_2: list):
    intersection = set(ids_1) & set(ids_2)
    if intersection:
        print(f"Same images found in {path_1} and {path_2}")
        print("Id:", intersection)


def split_coco(dataset_path: Path, images1_path: Path, out1_path: Path,
    images2_path: Path = None, out2_path: Path = None,
    images3_path: Path = None, out3_path: Path = None):

    # Read coco json
    with open(dataset_path) as f:
        data = json.load(f)

    data_images = data["images"]
    data_annots = data["annotations"]
    data_cats = data["categories"]
    images_ids = {i["file_name"]: i["id"] for i in data_images}
    
    # Get the ids of the images
    ids = [_get_ids(images1_path, images_ids)]
    if images2_path is not None:
        ids += [_get_ids(images2_path, images_ids)]
    if images3_path is not None:
        ids += [_get_ids(images3_path, images_ids)]
    
    # Check intersections
    if images2_path is not None:
        _check_intersection(images1_path, ids[0], images2_path, ids[1])
    if images3_path is not None:
        _check_intersection(images1_path, ids[0], images3_path, ids[2])
        _check_intersection(images2_path, ids[1], images3_path, ids[2])

    # Copy and filter annots
    new_images = [_filter(data_images, ids[0], "id")]
    new_annots = [_filter(data_annots, ids[0], "image_id")]
    if images2_path is not None:
        new_images += [_filter(data_images, ids[1], "id")]
        new_annots += [_filter(data_annots, ids[1], "image_id")]
    if images3_path is not None:
        new_images += [_filter(data_images, ids[2], "id")]
        new_annots += [_filter(data_annots, ids[2], "image_id")]

    # Save the new json datasets
    save_coco(out1_path, new_images, data_cats, new_annots)
    if images2_path is not None:
        save_coco(out2_path, new_images[1], data_cats, new_annots[1])
    if images3_path is not None:
        save_coco(out3_path, new_images[2], data_cats, new_annots[2])



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="""Split a coco dataset
    based on the images on a folder""")
    
    argparser.add_argument(
        'dataset',
        help="Path to the json dataset",
        type=Path
    )
    argparser.add_argument(
        '-i1',
        '--images-1',
        help="Path to the images",
        required=True,
        type=Path
    )
    argparser.add_argument(
        '-i2',
        '--images-2',
        help="Optional second path to images",
        required=False,
        type=Path
    )
    argparser.add_argument(
        '-i3',
        '--images-3',
        help="Optional third path to images",
        required=False,
        type=Path
    )
    argparser.add_argument(
        '-o1',
        '--output-1',
        help="Output json path",
        required=True,
        type=Path
    )
    argparser.add_argument(
        '-o2',
        '--output-2',
        help="Optional second output json path",
        required=False,
        type=Path
    )
    argparser.add_argument(
        '-o3',
        '--output-3',
        help="Optional third output json path",
        required=False,
        type=Path
    )
    args = argparser.parse_args()

    assert ((args.images_3 is None) or
            (args.images_2 is not None and args.images_3 is not None))
    
    assert ((args.images_2 is None) or
            (args.images_2 is not None and args.output_2 is not None))

    assert ((args.images_3 is None) or
            (args.images_3 is not None and args.output_3 is not None))

    split_coco(args.dataset, args.images_1, args.output_1,
        args.images_2, args.output_2, args.images_3, args.output_3)
