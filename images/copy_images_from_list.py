from pathlib import Path
from tqdm import tqdm
import argparse
import shutil


def copy_from_list(images_path: Path, dest_path: Path, list_file: Path, 
    is_absolute: bool):

    dest_path.mkdir(exist_ok=True, parents=True)
    
    files_to_copy = list_file.read_text().split("\n")
    pbar = tqdm(files_to_copy)
    for f in pbar:
        if f:
            src_path = str(images_path / f) if not is_absolute else f
            pbar.set_description(src_path)
            shutil.copy(src_path, dest_path)
        

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="""Copies the images
    on a list to another directory""")
    
    argparser.add_argument(
        "-i",
        "--images-path",
        help="Path to the images",
        type=Path,
        required=False
    )
    argparser.add_argument(
        "-d",
        "--dest-path",
        help="Destination path",
        type=Path,
        required=True
    )
    argparser.add_argument(
        "-l",
        "--list",
        help="File with the list of filenames to copy",
        type=Path,
        required=True
    )
    argparser.add_argument(
        "--is-absolute",
        help="The list file contains the absolute path to the files to copy",
        action="store_true"
    )

    args = argparser.parse_args()

    copy_from_list(args.images_path, args.dest_path, args.list, args.is_absolute)
