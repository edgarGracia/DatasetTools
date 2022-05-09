from pathlib import Path
import argparse
import json


def list_coco_images(dataset_path: Path, out_path: Path):

    with open(dataset_path) as f:
        data = json.load(f)

    data_images = data["images"]
    with open(out_path, "w") as f:
        for img in data_images:
            f.write(img["file_name"] + "\n")
        


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="""Saves on a file the
        filenames of the images on a COCO dataset""")
    
    argparser.add_argument(
        'dataset',
        help="Path to the json dataset",
        type=Path
    )
    argparser.add_argument(
        'output',
        help="Output file",
        type=Path
    )

    args = argparser.parse_args()

    list_coco_images(args.dataset, args.output)
